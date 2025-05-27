# Thoughts
- There is a decent improvement in predicting the direction of the stock price movement
- However, all models including HOPLS cannot predict return magnitude well (usually it is much too small)
- The higher-order interactions of HOPLS can capture systematic/market risk well, that is why the directional accuracy is fine. However the model might be 'averaging out' the idiosyncratic risk in each assets, which might be the cause as to why the magnitude of the return is not predicted well
- Factor models usually capture directional movement well but understate asset-specific shocks – only systematic risk is rewarded because idiosyncratic risk is diversified away
- (Ang, Hodrick, Xing, and Zhang, 2006):
  - > Stocks that have past high sensitivities to innovations in aggregate volatility have low average returns. We also find that stocks with past high idiosyncratic volatility have abysmally low returns, but this cannot be explained by exposure to aggregate volatility risk. The low returns earned by stocks with high exposure to systematic volatility risk and the low returns of stocks with high idiosyncratic volatility cannot be explained by the standard size, book-to-market, or momentum effects, and are not subsumed by liquidity or volume effects.
- HOPLS with low-rank apporixmation has a bias > because I am training it with data from 2000 - 2023 with 80/20 split, the model mostly captures time-invariant factors

# Ideas:
- Time-varying loadings  
  - [(Xu, 2022): Testing for time-varying factor loadings in high-dimensional factor models](https://www.tandfonline.com/doi/epdf/10.1080/07474938.2022.2074188?needAccess=true)
- Dynamic Factor Loadings with Instrumented PCA. See [(Kelly et al 2020)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2983919)
- Hybrid modelling apporach > model systematic part with HOPLS, learn the residuals with something else

# Ideas
- First dimension > no. of sliding windows
- number of windows, number of assets, cross section, time index inside each window

Y: number of windows, assets, returns