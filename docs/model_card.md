# Model Card

## Intended Use

Top-K item recommendation for e-commerce users based on implicit behavioral history.

## Dataset

RetailRocket e-commerce events from Kaggle. The full dataset is not committed to git.

## Target Definition

For offline evaluation, `addtocart` and `transaction` events are treated as relevant positive interactions. `view` events are used as weaker training signals.

## Validation Design

The project uses temporal train/validation/test splitting. The model is trained on earlier events and evaluated on later windows to better approximate production recommendation behavior.

## Known Limitations

- Offline metrics do not prove online business impact.
- User IDs are anonymous and may represent sessions rather than stable accounts.
- Cold-start handling is currently popularity-based.
- The first version optimizes for a clear portfolio-grade ML lifecycle, not maximum leaderboard performance.

