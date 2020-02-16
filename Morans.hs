{-# LANGUAGE TemplateHaskell #-}
{-# OPTIONS_GHC -Wall #-}
{-# OPTIONS_GHC -Wno-incomplete-patterns #-}
{-# OPTIONS_GHC -Wno-missing-signatures  #-}
{-# OPTIONS_GHC -Wno-type-defaults       #-}
{-# OPTIONS_GHC -Wno-name-shadowing      #-}

module Morans where

-- Requires files from http://yann.lecun.com/exdb/mnist/
import Codec.Compression.GZip (decompress)
import qualified Data.ByteString.Lazy as BS
import System.Environment
import System.Exit
import System.IO
import System.Random
import Data.Ord

import Control.Monad
import Data.List

import Hedgehog
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range

floatList :: MonadGen m => m [Float]
floatList = Gen.list (Range.linear 0 10)
                     (Gen.float (Range.linearFrac (-10) 10))

zLayerInput :: MonadGen m => m ([Float], ([Float], [[Float]]))
zLayerInput = do
  l1 <- floatList
  l2 <- floatList
  ll <- Gen.list (Range.linear 0 10) floatList

  return (l1, (l2, ll))

feedInput :: MonadGen m => m ([Float], [([Float], [[Float]])])
feedInput = do
  l1 <- floatList
  lt <- Gen.list (Range.linear 0 10)
                 ((,) <$> floatList
                  <*> (Gen.list (Range.linear 0 10) floatList))

  return (l1, lt)

prop_same_zLayer :: Property
prop_same_zLayer = property $ do
  (l1, t) <- forAll zLayerInput
  zLayer_new l1 t === zLayer l1 t

prop_same_feed :: Property
prop_same_feed = property $ do
  (l1, lt) <- forAll feedInput
  feed_new l1 lt === feed l1 lt

tests :: IO Bool
tests = checkSequential $$(discover)

gauss scale = do
  x1 <- randomIO
  x2 <- randomIO
  return $ scale * sqrt (-2 * log x1) * cos (2 * pi * x2)

newBrain :: [Int] -> IO [([Float], [[Float]])]
newBrain szs@(_:ts) = zip (flip replicate 1 <$> ts) <$>
  zipWithM (\m n -> replicateM n $ replicateM m $ gauss 0.01) szs ts

relu = max 0
relu' x | x < 0      = 0
        | otherwise  = 1

zLayer ::[Float] -> ([Float], [[Float]]) -> [Float]
zLayer as (bs, wvs) = zipWith (+) bs $ sum . zipWith (*) as <$> wvs

(.+) :: [Float] -> [Float] -> [Float]
(.+) = zipWith (+)

dot :: [Float] -> [Float] -> Float
dot v1 v2 = sum (zipWith (*) v1 v2)

(.*) :: [Float] -> [[Float]] -> [Float]
v .* m = dot v <$> m

type Vector = [Float]
type Matrix = [[Float]]

zLayer_new :: Vector -> (Vector, Matrix) -> Vector
zLayer_new as (bs, wvs) = bs .+ (as .* wvs)

feed :: [Float] -> [([Float], [[Float]])] -> [Float]
feed = foldl' (((relu <$>) . ) . zLayer)

feed_new :: [Float] -> [([Float], [[Float]])] -> [Float]
feed_new = foldl' (\v m -> relu <$> zLayer v m)

revaz xs = foldl' (\(avs@(av:_), zs) (bs, wms) -> let
  zs' = zLayer av (bs, wms) in ((relu <$> zs'):avs, zs':zs)) ([xs], [])

dCost a y | y == 1 && a >= y = 0
          | otherwise        = a - y

deltas xv yv layers = let
  (avs@(av:_), zv:zvs) = revaz xv layers
  delta0 = zipWith (*) (zipWith dCost av yv) (relu' <$> zv)
  in (reverse avs, f (transpose . snd <$> reverse layers) zvs [delta0]) where
    f _ [] dvs = dvs
    f (wm:wms) (zv:zvs) dvs@(dv:_) = f wms zvs $ (:dvs) $
      zipWith (*) [sum $ zipWith (*) row dv | row <- wm] (relu' <$> zv)

eta = 0.002

descend av dv = zipWith (-) av ((eta *) <$> dv)

learn xv yv layers = let (avs, dvs) = deltas xv yv layers
  in zip (zipWith descend (fst <$> layers) dvs) $
    zipWith3 (\wvs av dv -> zipWith (\wv d -> descend wv ((d*) <$> av)) wvs dv)
      (snd <$> layers) avs dvs

getImage s n = fromIntegral . BS.index s . (n*28^2 + 16 +) <$> [0..28^2 - 1]
getX     s n = (/ 256) <$> getImage s n
getLabel s n = fromIntegral $ BS.index s (n + 8)
getY     s n = fromIntegral . fromEnum . (getLabel s n ==) <$> [0..9]

render n = let s = " .:oO@" in s !! (fromIntegral n * length s `div` 256)

main = do
  as <- getArgs
  [trainI, trainL, testI, testL] <- mapM ((decompress  <$>) . BS.readFile)
    [ "train-images-idx3-ubyte.gz"
    , "train-labels-idx1-ubyte.gz"
    ,  "t10k-images-idx3-ubyte.gz"
    ,  "t10k-labels-idx1-ubyte.gz"
    ]

  when (as == ["samplesjs"]) $ do
    putStr $ unlines
      [ "var samples = " ++ show ((show . getImage testI) <$> [0..49]) ++ ";"
      , "function random_sample() {"
      , "  return samples[Math.floor(Math.random() * samples.length)];"
      , "}"
      ]
    exitSuccess

  hSetBuffering stderr LineBuffering
  let
    (pStr, pStrLn) = case as of
      ["print"] -> (hPutStr stderr, hPutStrLn stderr)
      _         -> (putStr, putStrLn)

  n <- (`mod` 10000) <$> randomIO
  pStr . unlines $
    take 28 $ take 28 <$> iterate (drop 28) (render <$> getImage testI n)

  b <- newBrain [784, 30, 10]
  let
    example = getX testI n
    bs = scanl (foldl' (\b n -> learn (getX trainI n) (getY trainL n) b)) b [
     [   0.. 999],
     [1000..2999],
     [3000..5999],
     [6000..9999]]
    smart = last bs
    cute d score = show d ++ ": " ++ replicate (round $ 70 * min 1 score) '+'
    bestOf = fst . maximumBy (comparing snd) . zip [0..]

  forM_ bs $ pStrLn . unlines . zipWith cute [0..9] . feed example

  pStrLn $ "best guess: " ++ show (bestOf $ feed example smart)

  let guesses = bestOf . (\n -> feed (getX testI n) smart) <$> [0..9999]
  let answers = getLabel testL <$> [0..9999]
  pStrLn $ show (sum $ fromEnum <$> zipWith (==) guesses answers) ++
    " / 10000"

  case as of
    ["print"] -> print smart
    _         -> return ()

newBrain_new :: [Int] -> IO [([Float], [[Float]])]
newBrain_new szs@(_:ts) = zip (flip replicate 1 <$> ts) <$>
  zipWithM (\m n -> replicateM n $ replicateM m $ gauss 0.01) szs ts

{-
prop_same_newBrain = property $ do
  l <- forAll (Gen.list (Range.linear 1 10)
                        (Gen.int (Range.linear 1 10)))
  newBrain_new l === newBrain l
-}
