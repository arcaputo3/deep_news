# deep_news
A trading algorithm which uses news and ticker data to predict the volatility and volume of ETFs.

We hypothesis that morning news and morning ticker data contain some predictive capabilities toward how the market will behave throughout the day.
As it turns out, our model finds some small predictive capability (~60% daily) for a handful of ETFs, and performed particularly well on larger ETFs such as QQQ and SPY.
From there, we attempted to trade a Relative Strength Index (RSI) strategy on ETFs that we predicted would have higher than usual volatility and volume. The reason for this is that
RSI strategies benefit from frequent fluctuation, and as it turns out outperform buy-and-hold given perfect predictions (ideal strategy). Our model achieves overall ~60% accuracy on out of fold predictions
and performs almost as well as the ideal strategy, however our daily returns are not statistically significantly superior to those of buy-and-hold.

Note that for copyright and intellectual property purposes, data and models have been excluded.
