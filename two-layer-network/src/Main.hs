module Main where

import qualified Control.Foldl as L
import Control.Lens (element, view, (^?))
import Control.Monad (ap, replicateM)
import Control.Monad.Trans.Class (MonadTrans (..))
import Control.Monad.Trans.Cont (ContT (..), evalContT)
import Control.Monad.Trans.State (StateT (..), evalStateT, gets, state)
import Data.Maybe (listToMaybe)
import Data.Random.Normal (normal)
import qualified Data.Set as Set
import Data.Text (Text, unpack)
import GHC.Exts (IsList (..))
import GHC.Float (float2Double)
import GHC.Generics (Generic)
import GHC.TypeLits (KnownNat, Nat)
import Graphics.Vega.VegaLite hiding (name, sample)
import qualified Pipes as P
import qualified Pipes.Concurrent as P
import qualified Pipes.Group as P
import qualified Pipes.Prelude as P
import System.Random (Random, RandomGen, getStdGen)
import Torch.Data.Internal (fromInput', toOutput', withBufferLifted)
import Torch.Typed hiding (DType, Device, shape, sin)
import Prelude hiding (filter, tanh)

-- | Use single precision for all tensor computations
type DType = 'Float

-- | Run all tensor computation on CPU
type Device = '( 'CPU, 0)

-- | Uncommend this to run on GPU instead
-- type Device = '( 'CUDA, 0)

-- | Compute the sine cardinal (sinc) function,
-- see https://mathworld.wolfram.com/SincFunction.html
sinc :: Floating a => a -> a
sinc a = sin a / a

-- | Compute the sine cardinal (sinc) function and add normally distributed noise
-- of strength epsilon
noisySinc :: (Floating a, Random a, RandomGen g) => a -> a -> g -> (a, g)
noisySinc eps a g = let (noise, g') = normal g in (sinc a + eps * noise, g')

-- | Datatype to represent a dataset of sine cardinal (sinc) inputs and outputs
data SincData = SincData {name :: Text, unSincData :: [(Float, Float)]} deriving (Eq, Ord)

-- | Create a dataset of noisy sine cardinal (sinc) values of a desired size
mkSincData :: (RandomGen g, Monad m) => Text -> Int -> StateT g m SincData
mkSincData name size =
  let next = do
        x <- (* 20) <$> state normal
        y <- state (noisySinc 0.05 x)
        pure (x, y)
   in SincData name <$> replicateM size next

-- | 'Dataset' instance used for streaming sine cardinal (sinc) examples
instance Dataset IO SincData Int (Float, Float) where
  getItem (SincData _ d) k = maybe (fail "invalid key") pure $ d ^? element k
  keys (SincData _ d) = Set.fromList [0 .. Prelude.length d -1]

-- | Datatype to represent the model parameters.
-- The datatype is parameterized over the number of input, output, and hidden dimensions.
-- Each 'Linear' datatype holds internal weight tensors for weight and bias.
data TwoLayerNet (numIn :: Nat) (numOut :: Nat) (numHidden :: Nat) = TwoLayerNet
  { linear1 :: Linear numIn numHidden DType Device, -- first linear layer
    linear2 :: Linear numHidden numOut DType Device -- second linear layer
  }
  deriving (Show, Generic, Parameterized)

-- | 'HasForward' instance used to define the batched forward pass of the model
instance
  HasForward
    (TwoLayerNet numIn numOut numHidden)
    (Tensor Device DType '[batchSize, numIn])
    (Tensor Device DType '[batchSize, numOut])
  where
  forward TwoLayerNet {..} =
    -- call the linear forward function on the 'Linear' datatypes
    -- and sandwich a 'tanh' activation function in between
    forward linear2 . tanh . forward linear1
  forwardStoch = (pure .) . forward

-- | Train the model for one epoch
train ::
  _ =>
  -- | initial model datatype holding the weights
  model ->
  -- | initial optimizer, e.g. Adam
  optim ->
  -- | learning rate, 'LearningRate device dtype' is a type alias for 'Tensor device dtype '[]'
  LearningRate device dtype ->
  -- | stream of training examples consisting of inputs, outputs, and an iteration counter
  P.ListT IO (Tensor device dtype shape, Tensor device dtype shape, Int) ->
  -- | final model and optimizer
  IO (model, optim)
train model optim learningRate examples =
  let -- training step function
      step (model, optim) (x, y, _iter) = do
        let -- compute predicted y' by passing x to the model
            y' = forward model x
            -- compute the loss from the predicted values y' and the true values y;
            -- the loss function is the reduced Mean Squared Error (MSE),
            -- i.e. the average squared difference between the predicted values and the true values
            loss = mseLoss @ 'ReduceMean y' y
        -- compute gradient of the loss with respect to all the learnable parameters of the model
        -- and update the weights using the optimizer.
        runStep model optim loss learningRate
      -- initial training state
      init = pure (model, optim)
      -- trivial extraction function
      done = pure
   in -- training is a fold over the 'examples' stream
      P.foldM step init done . P.enumerate $ examples

-- | Evaluate the model
evaluate ::
  forall m model device batchSize shape.
  _ =>
  -- | model to be evaluated
  model ->
  -- | stream of evaluation examples consisting of inputs, outputs, and an iteration counter
  P.ListT m (Tensor device DType (batchSize ': shape), Tensor device DType (batchSize ': shape), Int) ->
  m Float
evaluate model examples =
  let -- evaluation step function
      step (loss, _) (x, y, iter) = do
        let y' = forward model x
            loss' = toFloat $ mseLoss @ 'ReduceMean y' y
        pure (loss + loss', iter)
      -- initial evaluation state
      init = pure (0, 0)
      -- calculate the average loss per example and return
      done (_, 0) = pure 0
      done (loss, iter) =
        let scale x = x / (fromInteger . toInteger $ iter + natValI @batchSize)
         in pure $ scale loss
   in -- like training, evaluation is a fold over the 'examples' stream
      P.foldM step init done . P.enumerate $ examples

-- | Run inference using a trained model
infer ::
  _ =>
  -- | model to use for inference
  model ->
  -- | stream of inputs to run inference on
  P.ListT m (Tensor device DType shape) ->
  -- | stream of input and output pairs
  P.ListT m (Tensor device DType shape, Tensor device DType shape)
infer model (P.Select inputs) = P.Select $ inputs P.>-> P.map (\x -> (x, forward model x))

-- | Batch size
type BatchSize = 100

-- | Number of hidden layers
type NumHidden = 100

-- | Main program
main :: IO ()
main = do
  model <-
    -- initialize the model
    TwoLayerNet @1 @1 @NumHidden
      -- randomly initialize the weights and biases of the first linear layer
      <$> sample LinearSpec
      -- randomly initialize the weights and biases of the second linear layer
      <*> sample LinearSpec

  let -- use the stochastic gradient descent optimization algorithm where `params <- params - learningRate * gradients`
      -- optim = mkGD

      -- use the Adam optimization algorithm, see https://arxiv.org/abs/1412.6980
      optim = mkAdam 0 0.9 0.999 (flattenParameters model)

  let -- learning rate(s)
      maxLearningRate = 1e-2
      finalLearningRate = 1e-4

      -- total number of epochs
      numEpochs = 100
      numWarmupEpochs = 10
      numCooldownEpochs = 10

      -- single-cycle learning rate schedule, see for instance https://arxiv.org/abs/1803.09820
      learningRateSchedule epoch
        | epoch <= 0 = 0.0
        | 0 < epoch && epoch <= numWarmupEpochs =
          let a :: Float = fromIntegral epoch / fromIntegral numWarmupEpochs
           in mulScalar a maxLearningRate
        | numWarmupEpochs < epoch && epoch < numEpochs - numCooldownEpochs =
          let a :: Float =
                fromIntegral (numEpochs - numCooldownEpochs - epoch)
                  / fromIntegral (numEpochs - numCooldownEpochs - numWarmupEpochs)
           in mulScalar a maxLearningRate + mulScalar (1 - a) finalLearningRate
        | otherwise = finalLearningRate

  (trainingData, evaluationData, options) <-
    getStdGen
      >>= evalStateT
        ( (,,)
            -- create a dataset of 10000 unique training examples
            <$> mkSincData "training" 10000
              -- create a dataset of 500 unique evaluation examples
              <*> mkSincData "evaluation" 500
              -- configure the data loader for random shuffling
              <*> gets (\g -> (datasetOpts 1) {shuffle = Shuffle g})
        )

  let -- generate statistics and plots for each epoch
      stats model learningRate trainingLosses evaluationLosses learningRates options = do
        learningRates <- pure $ toFloat learningRate : learningRates
        (trainingLoss, options) <- evaluate' model options trainingData
        (evaluationLoss, options) <- evaluate' model options evaluationData
        trainingLosses <- pure $ trainingLoss : trainingLosses
        evaluationLosses <- pure $ evaluationLoss : evaluationLosses
        plot "plot.html" (infer' model) trainingLosses evaluationLosses learningRates evaluationData
        pure (trainingLosses, evaluationLosses, learningRates, options)
      -- program step function
      step (model, optim, trainingLosses, evaluationLosses, learningRates, options) epoch = do
        let learningRate = learningRateSchedule epoch
        (model, optim, options) <-
          -- train the model
          train' model optim learningRate options trainingData
        (trainingLosses, evaluationLosses, learningRates, options) <-
          -- calculate epoch statistics
          stats model learningRate trainingLosses evaluationLosses learningRates options
        pure (model, optim, trainingLosses, evaluationLosses, learningRates, options)
      -- initial program state
      init = do
        let learningRate = learningRateSchedule 0
        (trainingLosses, evaluationLosses, learningRates, options) <-
          -- calculate initial statistics
          stats model learningRate [] [] [] options
        pure (model, optim, trainingLosses, evaluationLosses, learningRates, options)
      -- just keep the final model
      done (model, _, _, _, _, _) = pure model
  -- the whole program is a fold over epochs
  model <- evalContT . P.foldM step init done . P.each $ [1 .. numEpochs]
  -- save the model weights to a file and end program
  save (hmap' ToDependent . flattenParameters $ model) "model.pt"

train' :: _ => model -> optim -> LearningRate device dtype -> DatasetOptions -> dataset -> ContT r IO (model, optim, DatasetOptions)
train' model optim learningRate options sincData = do
  (examples, shuffle) <- streamFromMap options sincData
  (model, optim) <-
    lift . train model optim learningRate
      =<< Main.collate @BatchSize 1 (P.Select $ P.zip (P.enumerate examples) (P.each [0 ..]))
  pure (model, optim, options {shuffle = shuffle})

evaluate' :: _ => model -> DatasetOptions -> SincData -> ContT r IO (Float, DatasetOptions)
evaluate' model options sincData = do
  (examples, shuffle) <- streamFromMap options sincData
  loss <-
    lift . evaluate model
      =<< Main.collate @BatchSize @Device 1 (P.Select $ P.zip (P.enumerate examples) (P.each [0 ..]))
  lift . putStrLn $ "Average " <> unpack (name sincData) <> " loss: " <> show loss
  pure (loss, options {shuffle = shuffle})

infer' :: _ => model -> [Float] -> ContT r IO [(Float, Float)]
infer' model xs =
  lift . P.toListM . P.enumerate . disperse . infer model
    =<< collate' @BatchSize @Device @DType 1 (P.Select . P.each $ xs)

plot ::
  _ =>
  -- | output file path
  FilePath ->
  -- | inference continuation
  ([Float] -> ContT r IO [(Float, Float)]) ->
  -- | training losses
  [Float] ->
  -- | evaluation losses
  [Float] ->
  -- | learning rates
  [Float] ->
  -- | evaluation data
  SincData ->
  -- | returned continuation
  ContT r IO ()
plot file infer trainingLosses evaluationLosses learningRates sincData = do
  let xs = [-20, -19.95 .. 20]
      epochs = [0, 1 ..]
      mkDataRow x y =
        dataRow
          [ ("x", Number . float2Double $ x),
            ("y", Number . float2Double $ y)
          ]
  predictionData <- do
    rawRows <- infer xs
    pure . dataFromRows [] . foldr (uncurry mkDataRow) [] $ rawRows
  let evaluationData = dataFromRows [] . foldr (uncurry mkDataRow) [] . unSincData $ sincData
  let targetData = dataFromRows [] . foldr (mkDataRow `ap` sinc) [] $ xs
  let lossData =
        let mkDataRows title losses =
              zipWith
                ( \epoch loss ->
                    dataRow
                      [ ("epoch", Number . fromInteger $ epoch),
                        ("title", Str title),
                        ("loss", Number . float2Double $ loss)
                      ]
                )
                epochs
                (reverse losses)
         in dataFromRows [] . foldr id [] $
              mkDataRows "training" trainingLosses
                <> mkDataRows "evaluation" evaluationLosses
  let learningRateData =
        let dataRows =
              zipWith
                ( \epoch learningRate ->
                    dataRow
                      [ ("epoch", Number . fromInteger $ epoch),
                        ("learning_rate", Number . float2Double $ learningRate)
                      ]
                )
                epochs
                (reverse learningRates)
         in dataFromRows [] . foldr id [] $ dataRows
  lift . toHtmlFile file . mkVegaLite . datasets $
    [ ("prediction", predictionData),
      ("evaluation", evaluationData),
      ("target", targetData),
      ("losses", lossData),
      ("learning_rates", learningRateData)
    ]

collate ::
  forall batchSize device dtype r m.
  (KnownNat batchSize, KnownDevice device, ComputeHaskellType dtype ~ Float, MonadBaseControl IO m) =>
  Int ->
  P.ListT m ((Float, Float), Int) ->
  ContT r m (P.ListT m (Tensor device dtype '[batchSize, 1], Tensor device dtype '[batchSize, 1], Int))
collate n = Main.bufferedCollate (P.bounded n) (natValI @batchSize) f
  where
    f exs =
      let (xys, iters) = unzip exs
          (xs, ys) = unzip xys
       in (,,) <$> fromList (pure <$> xs) <*> fromList (pure <$> ys) <*> listToMaybe iters

disperse ::
  forall batchSize device dtype m.
  (KnownNat batchSize, KnownDevice device, ComputeHaskellType dtype ~ Float, Functor m) =>
  P.ListT m (Tensor device dtype '[batchSize, 1], Tensor device dtype '[batchSize, 1]) ->
  P.ListT m (Float, Float)
disperse (P.Select xs) = P.Select $ P.for xs (P.each . uncurry f)
  where
    f x y = do
      xs <- zip (toList . Just $ x) (toList . Just $ y)
      case xs of
        ([x], [y]) -> pure (x, y)
        _ -> mempty

collate' ::
  forall batchSize device dtype r m.
  (KnownNat batchSize, KnownDevice device, ComputeHaskellType dtype ~ Float, MonadBaseControl IO m) =>
  Int ->
  P.ListT m Float ->
  ContT r m (P.ListT m (Tensor device dtype '[batchSize, 1]))
collate' n = Main.bufferedCollate (P.bounded n) (natValI @batchSize) f
  where
    f xs = fromList (pure <$> xs)

bufferedCollate ::
  forall a b r m.
  MonadBaseControl IO m =>
  P.Buffer b ->
  Int ->
  ([a] -> Maybe b) ->
  P.ListT m a ->
  ContT r m (P.ListT m b)
bufferedCollate buffer batchSize f as = ContT $ \g ->
  snd
    <$> withBufferLifted
      buffer
      fOutput
      (g . P.Select . fromInput')
  where
    fOutput output = P.runEffect $ (P.>-> toOutput' output) . (P.>-> P.mapMaybe f) . L.purely P.folds L.list . view (P.chunksOf batchSize) . P.enumerate $ as

mkVegaLite :: Data -> VegaLite
mkVegaLite dataset =
  let -- width and height of the individual plots (in pixels)
      w = width 700
      h = height 350

      encOverview =
        encoding
          . position X [PName "x", PmType Quantitative]
          . position Y [PName "y", PmType Quantitative, PScale scaleOptsOverview]
      scaleOptsOverview = [SDomain (DNumbers [-2, 2]), SNice (IsNice False)]
      transOverview =
        transform
          . filter (FRange "x" (NumberRange (-20) 20))
          . filter (FRange "y" (NumberRange (-2) 2))
      target =
        asSpec
          [ dataFromSource "target" [],
            transOverview [],
            mark Line [MStrokeWidth 0.5, MStroke "black"]
          ]
      evaluation =
        asSpec
          [ dataFromSource "evaluation" [],
            transOverview [],
            mark Point [MSize 5, MStroke "black"]
          ]
      prediction =
        asSpec
          [ dataFromSource "prediction" [],
            transOverview [],
            mark Line []
          ]
      overview =
        asSpec
          [ layer [target, evaluation, prediction],
            encOverview [],
            w,
            h
          ]

      encLosses =
        encoding
          . position X [PName "epoch", PmType Quantitative, PScale scaleOptsEpoch]
          . position Y [PName "loss", PmType Quantitative, PScale [SType ScLog]]
          . color [MName "title", MmType Nominal, MLegend [LTitle "", LOrient LOBottom]]
      scaleOptsEpoch = [SDomain (DNumbers [0, 100]), SNice (IsNice False)]
      losses =
        asSpec
          [ dataFromSource "losses" [],
            encLosses [],
            mark Line [],
            w,
            h
          ]

      encLearningRate =
        encoding
          . position X [PName "epoch", PmType Quantitative, PScale scaleOptsEpoch]
          . position Y [PName "learning_rate", PTitle "learning rate", PmType Quantitative]
      learningRates =
        asSpec
          [ dataFromSource "learning_rates" [],
            encLearningRate [],
            mark Line [],
            w,
            h
          ]
   in toVegaLite
        [ dataset,
          vConcat [overview, losses, learningRates]
        ]
