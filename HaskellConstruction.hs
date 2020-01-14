{-# LANGUAGE RankNTypes, MultiParamTypeClasses, GADTs, PolyKinds, DataKinds, ScopedTypeVariables, FlexibleInstances #-}
module HaskellConstruction where

import Data.List

type Vector r  = [r]
type Matrix r  = [Vector r]


data Network = Network { getNet :: (forall r . (Num r, ReLU r, Fractional r, Show r) => Vector r -> r) }

cartProd :: [a] -> [[a]] -> [[a]]
cartProd xs ys = [ x:y | y <- ys, x <- xs]

cartProds :: [[a]] -> [[a]]
cartProds xls = foldr cartProd [[]] xls

convert :: (Real a, Fractional b) => a -> b
convert = fromRational . toRational

class ReLU t where
  relu :: t -> t
  
instance ReLU Double where
  relu x = if x > 0 then x else 0

nmin2 :: (Num r, ReLU r, Fractional r) => r -> r -> r
nmin2 x y = 0.5 * (sum $ zipWith (*) [1, -1, -1, -1] $ map relu [x + y, -1 * x - y, x - y, y - x] )

nmin :: (Num r, ReLU r, Fractional r) => [r] -> r
nmin [x] = x
nmin xr = foldr1 nmin2 xr

reluC :: (Num t, ReLU t) => t -> t -> t
reluC b v = b - relu (b - v)

bump :: forall r . (Show r, Real r, Num r, RealFrac r, ReLU r, Fractional r, Floating r) => Int ->  (Vector r,Vector r) -> Network
bump big_m (lb, ub) = Network $ \x -> relu $ nmin $ map (\t -> reluC 1 (convert l * convert big_m * t + 1) ) $ concat
                                $ [ [xk - convert ilkm , convert iukm - xk ] | (xk, ilkm, iukm) <- zip3 x lb ub ]
  where l = 2.0 ** (convert $ ceiling (logBase 2 $ 2 * m) + 1)
        m = fromIntegral $ length lb

lem44 :: forall r . (Real r, RealFrac r, Ord r, ReLU r, Floating r, Show r)
         => (Vector r -> Vector r -> (r, r)) -- ^ the function represented as closed box to min and max in that region
         -> Vector r -- ^ lower bound vector
         -> Vector r -- ^ upper bound vector
         -> r -- ^ epsilon
         -> r -- ^ delta
         -> Int -- ^ number of slices
         -> Int -- ^ M s.t. if |x - y| <= 1/m, then |f(y)−f(x)| ≤ δ
         -> [Network]
lem44 f l u epsilon delta n m = 
  [ Network $ \x -> reluC 1 $ sum $ map (\(a,b) -> getNet (bump num_grid_points (a,b)) x) bump_boxes | bump_boxes <- slice_boxes ]
  where xi0 = fst $ f l u
        xiN = snd $ f l u
        height = xiN - xi0
        slice_height = height / fromIntegral n
        
        max_grid_width = 1.0 / fromIntegral m
        
        num_grid_points :: Int
        num_grid_points = maximum $ zipWith (\lv uv -> ceiling $ (uv - lv) / max_grid_width) l u

        grid_points =  cartProds $ map (\lv -> [fromIntegral k * max_grid_width + lv | k <- [0..num_grid_points]]) l
                       
        isBox lb ub = all (uncurry (<=)) (zip lb ub) && any (uncurry (<)) (zip lb ub)
        
        mins = [(lb, ub, fst $ f lb ub) | lb <- grid_points, ub <- grid_points, isBox lb ub]

        slice_heights = [ fromIntegral k * slice_height + xi0 | k <- [1..n]]

        slice_boxes :: [[(Vector r,Vector r)]]
        slice_boxes = [ map (\(a,b,_) -> (a,b)) $ filter (\(lb,ub, mn) -> mn >= h + 0.5 * delta) mins | h <- slice_heights ]


        
buildProvableNet :: (Real r, RealFrac r, Ord r, ReLU r, Floating r, Show r)
                    => (Vector r -> Vector r -> (r, r)) -- ^ the function represented as closed box to min and max in that region
                    -> Vector r -- ^ lower bound vector of the region
                    -> Vector r -- ^ upper bound vector  of the region
                    -> r -- ^ epsilon
                    -> r -- ^ delta
                    -> Int -- ^ M s.t. if |x - y| <= 1/m, then |f(y)−f(x)| ≤ δ
                    -> Network
buildProvableNet f l u epsilon delta m = Network $ \x ->
  (convert xi0) + 0.5 * (convert delta') * (sum $ map (($ x) . getNet) nets)
  where xi0 = fst $ f l u
        xiN = snd $ f l u
        n = ceiling $ 2 * height / delta
        delta' = 2 * (height / convert n)
        
        height = xiN - xi0
        
        nets = lem44 f l u epsilon delta' n m

data Interval a = Interval a a deriving Show

instance (Ord a, Num a) => Num (Interval a) where
  (+) (Interval a b) (Interval c d) = Interval (a + c) (b + d)
  negate (Interval a b) = Interval (-b) (-a)
  abs (Interval a b) = Interval (maximum [a, 0, -b]) (maximum [a,b, -a, -b])
  (*) (Interval a b) (Interval c d) = Interval (minimum combs) (maximum combs)
    where combs = [a * c, a * d, b * c, b * d]
  fromInteger a = Interval f f where f = fromInteger a
  signum (Interval a b) = Interval (signum a) (signum b)

instance ReLU a => ReLU (Interval a) where
  relu (Interval a b) = Interval (relu a) (relu b)

instance (Ord a, Fractional a) => Fractional (Interval a) where
  fromRational a = Interval f f where f = fromRational a
  recip (Interval a b) = if a >= 0 || b <= 0
                         then Interval (1 / b) (1 / a) 
                         else Interval (1 / a) (1 / b) 

p2 x = x * x
p4 x = let y = p2 x in y * y


test = do
  let f :: Vector Double -> Double
      f x = sum $ map
            (\x -> let s = 2 * x in p2 s - p4 s)
            x

      fmax :: Vector Double -> Vector Double -> (Double, Double)
      fmax [lb] [ub] = (minimum fs, maximum fs)
        where fs = [flb, fub] ++concatMap relevance extreema
              flb = f [lb]
              fub = f [ub]
              extreema = [- 1/(2 * (2 ** 0.5)), 0 , 1/(2*  (2 ** 0.5))]
              relevance e = if lb <= e && e <= ub then [f [e]] else []
  
      epsilon = 0.36
      delta = 0.2

      xlb = [0]
      xub = [1]
      
      net :: Network
      net = buildProvableNet fmax xlb xub epsilon delta 100
      
      pt = 0.25
      
      a :: Interval Double
      a@(Interval al au) = uncurry Interval $ fmax [pt] [(pt + epsilon)]
      
      i :: Interval Double
      i@(Interval il iu) = getNet net [Interval pt (pt + epsilon)]

  putStrLn $ "Epsilon: " ++ show epsilon
  putStrLn $ "Delta: " ++ show delta

  putStrLn $ "lb: " ++ show pt
  putStrLn $ "ub: " ++ show (pt + epsilon)
  
  putStrLn $ "\nf(lb,ub): " ++ show a
  putStrLn $ "(f(lb,ub).lb + delta, f(lb,ub).ub - delta): " ++ show (Interval (al + delta) (au -  delta))
  putStrLn $ "certified net#(lb,ub): " ++ show i
  putStrLn $ "(f(lb,ub).lb - delta, f(lb,ub).ub + delta): " ++ show (Interval (al - delta) (au + delta))

  putStrLn $ "(Net(lb),Net(ub)): " ++ show (Interval (getNet net [pt]) (getNet net [pt + epsilon]))
