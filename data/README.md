# Data dictionary

All data is synthetic and reproducible from the scripts (`make data`).

## raw/reviews.csv (one row per review)
| column | meaning |
|---|---|
| review_id, restaurant_id, restaurant_name, cuisine, price_tier | identifiers / restaurant attributes |
| channel | `delivery` or `dine_in` |
| rating | 1-5 stars |
| text | review text |
| date | review date (2026) |
| model | which generator wrote the text (`offline-templates` or a Claude model id) |
| true_complaints, true_praises | **ground truth, hidden from the NLP pipeline**; pipe-separated; used only to score it |

## raw/experiment_users.parquet (one row per user in the test)
| column | meaning |
|---|---|
| user_id, variant | `control` / `treatment` (Review Highlights) |
| first_exposure_day | 0-27, day the user first reached a restaurant page in the test |
| user_type | `new` or `returning` |
| platform, city_tier | pre-registered segment dimensions |
| pre_period_orders | orders in the 4 weeks before the test (CUPED covariate) |
| orders_7d, revenue_7d, refunds_7d | within 7 days of first exposure |
| converted_7d | **primary metric**: ≥1 order within 7 days |
| full_7d_window | True if the user's 7-day window closed before the test ended |

## raw/experiment_sessions.parquet (one row per user-day visit)
`user_id, day, days_since_exposure, variant, user_type, platform, city_tier, ordered, order_value, refunded`

## processed/
* `reviews_enriched.csv`: reviews + VADER/supervised sentiment + `tag_*` complaint/praise flags
* `seed_robustness.csv`: key results across 20 simulation seeds
