-- Requires files from http://yann.lecun.com/exdb/mnist/
import           Codec.Compression.GZip         ( decompress )
import qualified Data.ByteString.Lazy          as BS
import           System.Environment
import           System.Exit
import           System.IO
import           System.Random
import           Data.Ord
import           GHC.Int                        ( Int64 )

import           Control.Monad
import           Data.List
import           Data.Foldable                  ( for_ )

type Biases = [Float]
type Weights = [[Float]]
type NeuralNet = [(Biases, Weights)]

gauss :: Float -> IO Float
gauss scale = do
  x1 <- randomIO
  x2 <- randomIO
  return $ scale * sqrt (-2 * log x1) * cos (2 * pi * x2)

newBrain :: [Int] -> IO NeuralNet
newBrain sizes =
  flip mapM sizePairs $ \(n, m) -> do
      let v = constVector m
      m <- gaussMatrix n m
      return (v, m)

  where gaussMatrix = \m n -> replicateM n $ replicateM m $ gauss 0.01
        constVector n = replicate n 1
        sizePairs = zip sizes (tail sizes)

-- activation function
relu :: Float -> Float
relu = max 0

-- derivative of activation function
relu' :: (Ord a, Num a, Num p) => a -> p
relu' x | x < 0     = 0
        | otherwise = 1

zLayer :: [Float] -> ([Float], [[Float]]) -> [Float]
zLayer as (bs, wvs) = bs .+ (as .* wvs)
  where x .+ y = zipWith (+) x y

(.*) :: [Float] -> [[Float]] -> [Float]
x .* y = dot x <$> y

dot :: [Float] -> [Float] -> Float
dot x y = sum (zipWith (*) x y)

feed :: [Float] -> NeuralNet -> [Float]
feed = foldl' (\v layer -> map relu (zLayer v layer))

-- xs: vector of inputs
-- Returns a list of (weighted inputs, activations) of each layer,
-- from last layer to first.
revaz
  :: Foldable t => [Float] -> t ([Float], [[Float]]) -> ([[Float]], [[Float]])
revaz xs = foldl'
  (\(avs@(av : _), zs) (bs, wms) ->
    let zs' = zLayer av (bs, wms) in ((relu <$> zs') : avs, zs' : zs)
  )
  ([xs], [])

dCost :: (Num p, Ord p) => p -> p -> p
dCost a y | y == 1 && a >= y = 0
          | otherwise        = a - y

-- xv: vector of inputs
-- yv: vector of desired outputs
-- Returns list of (activations, deltas) of each layer in order.
deltas :: [Float] -> [Float] -> NeuralNet -> ([[Float]], [[Float]])
deltas xv yv layers =
  let (avs@(av : _), zv : zvs) = revaz xv layers
      delta0 = zipWith (*) (zipWith dCost av yv) (relu' <$> zv)
      weights = snd <$> layers
  in  (reverse avs, f ((transpose . reverse) weights) zvs [delta0]) where
  f _          []         dvs          = dvs
  f (wm : wms) (zv : zvs) dvs@(dv : _) =
    f wms zvs (zipWith (*) (dv .* wm) (relu' <$> zv) : dvs)

eta :: Float
eta = 0.002

descend :: [Float] -> [Float] -> [Float]
descend av dv = av .- (eta ..* dv)
  where x .- y = zipWith (-) x y

(..*) :: Float -> [Float] -> [Float]
lambda ..* x = map (lambda *) x

learn :: [Float] -> [Float] -> NeuralNet -> NeuralNet
learn xv yv layers =
  let (avs, dvs) = deltas xv yv layers
      weights = snd <$> layers
      biases  = fst <$> layers

      newWeights = zipWith3
        (\wvs av dv -> zipWith (\wv d -> descend wv (d ..* av)) wvs dv)
        weights
        avs
        dvs

      newBiases = zipWith descend biases dvs

  in  zip newBiases newWeights

getImage :: Num b => BS.ByteString -> Int64 -> [b]
getImage s n =
  fromIntegral . BS.index s . (n * 28 * 28 + 16 +) <$> [0 .. 28 * 28 - 1]

getX :: Fractional b => BS.ByteString -> Int64 -> [b]
getX s n = (/ 256) <$> getImage s n

getLabel :: Num b => BS.ByteString -> Int64 -> b
getLabel s n = fromIntegral $ BS.index s (n + 8)

getY :: Num b => BS.ByteString -> Int64 -> [b]
getY s n = fromIntegral . fromEnum . (getLabel s n ==) <$> [0 .. 9]

render :: Integral a => a -> Char
render n = let s = " .:oO@" in s !! (fromIntegral n * length s `div` 256)

main :: IO ()
main = do
  as <- getArgs

  let loadZip filename = decompress <$> BS.readFile filename

  trainI <- loadZip "train-images-idx3-ubyte.gz"
  trainL <- loadZip "train-labels-idx1-ubyte.gz"
  testI  <- loadZip "t10k-images-idx3-ubyte.gz"
  testL  <- loadZip "t10k-labels-idx1-ubyte.gz"

  when (as == ["samplesjs"]) $ do
    putStr $ unlines
      [ "var samples = " ++ show (show . getImage testI <$> [0 .. 49]) ++ ";"
      , "function random_sample() {"
      , "  return samples[Math.floor(Math.random() * samples.length)];"
      , "}"
      ]
    exitSuccess

  hSetBuffering stderr LineBuffering
  let (pStr, pStrLn) = case as of
        ["print"] -> (hPutStr stderr, hPutStrLn stderr)
        _         -> (putStr, putStrLn)

  n <- (`mod` 10000) <$> randomIO
  pStr . unlines $ take 28 $ take 28 <$> iterate (drop 28)
                                                 (render <$> getImage testI n)

  b <- newBrain [784, 30, 10]
  let example = getX testI n
      bs = scanl (foldl' (\b n -> learn (getX trainI n) (getY trainL n) b))
                 b
                 [[0 .. 999], [1000 .. 2999], [3000 .. 5999], [6000 .. 9999]]
      smart = last bs
      cute d score = show d ++ ": " ++ replicate (round $ 70 * min 1 score) '+'
      bestOf = fst . maximumBy (comparing snd) . zip [0 ..]

  for_ bs $ pStrLn . unlines . zipWith cute [0 .. 9] . feed example

  pStrLn $ "best guess: " ++ show (bestOf $ feed example smart)

  let guesses = bestOf . (\n -> feed (getX testI n) smart) <$> [0 .. 9999]
  let answers = getLabel testL <$> [0 .. 9999]
  pStrLn $ show (sum $ fromEnum <$> zipWith (==) guesses answers) ++ " / 10000"

  case as of
    ["print"] -> print smart
    _         -> return ()
