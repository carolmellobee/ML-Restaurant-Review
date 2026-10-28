# Recommendation: Review Highlights & marketing direction

**To:** Marketing & Product leadership · **From:** Data Science · **Test:** `review_highlights_v1` (28 days)

> All data in this project is simulated. The reasoning transfers; the specific numbers are illustrative.

## The one-line answer

**Ship Review Highlights to new users only, move review-based social proof into acquisition marketing, keep it out of retention, and fix delivery reliability before anyone advertises speed.**

## What the test showed

| | 7-day conversion (headline) | Long run (7+ days after first exposure) |
|---|---|---|
| New users | +11% | **+7.6%** (95% CI +2.9% to +12.3%) |
| Returning users | +3.7% | **−5.6%** (95% CI −8.7% to −2.5%) |
| Everyone | **+7.2%** (CI +4.7% to +9.7%, p < 0.001) | **−1.4%** (CI −4.0% to +1.2%) |

The headline +7.2% is real but misleading. Most of it is a **novelty bump** that fades within a
few days. Once it's gone, the card **helps new users and hurts returning users**, and returning
users place ~69% of long-run orders. A full rollout would most likely lose orders.

The daily dashboard showed **+37% on day 1**. Had we stopped then (as unplanned peeking invites),
we would have reported a number five times too large. Our pre-registered weekly O'Brien-Fleming
looks held the false-positive rate at 5%; daily peeking would have pushed it to ~29%.

## Options, sized on the steady-state effect

Assumes 1,000,000 monthly orders from comparable traffic and the test's average order value (≈R$57).
Replace both with real figures.

| Option | Steady-state orders | 95% CI | Orders / month | Revenue / month |
|---|---|---|---|---|
| A. Ship to everyone | −1.4% | −4.0% to +1.2% | −14k | −R$0.8M |
| **B. Ship to new users only** | **+2.4%** | **+0.9% to +3.9%** | **+24k** | **+R$1.4M** |
| C. Don't ship | 0 | — | 0 | 0 |

Option B is the only one whose interval excludes a loss.

## Recommendations

1. **Product.** Show Review Highlights to users with fewer than 2 lifetime orders; switch it off
   after the 2nd order. Keep a 10% holdout of new users for 8 weeks to confirm the steady-state lift.

2. **Marketing direction: social proof is an acquisition lever, not a retention lever.**
   * Use "what reviewers love" (taste and friendly staff are the most-praised aspects) on SEO
     restaurant pages, paid-social creative, first-order onboarding emails and app-store assets.
   * Keep it out of CRM to existing customers. Retention messaging should be about *their* history:
     reorder favorites, loyalty perks, restaurants similar to ones they rate highly.
   * Each channel is its own hypothesis. Test with a power analysis first and ≥3 weeks runtime.

3. **Don't promise speed. Fix it.** Delivery delay is the costliest complaint in the reviews:
   present in ~41% of 1-2★ reviews and associated with ≈1.2★ lost when it occurs (the most stars
   lost per 100 reviews of any complaint). A "fast delivery" campaign would acquire customers
   straight into the #1 cause of 1★ reviews. Instead, with Ops: **honest ETAs + an automatic
   late-delivery credit**, tested on returning users with 30-day retention as the primary metric.

4. **Ignore the city-tier "finding."** Tier 1 looked stronger than tier 2, and the interaction
   test even crossed p < 0.05. The simulation has no city effect at all; the gap reversed in later
   data and has no mechanism. Regional strategy needs its own test.

5. **Process for every future test.** Pre-registered sequential looks (no ad-hoc stopping);
   minimum 3 weeks for UI changes; decide on the post-novelty effect; judge segments with
   interaction tests, not "significant here, not there".

## What would change this recommendation

* The holdout shows the new-user lift decaying toward zero after 8 weeks → feature isn't worth maintaining.
* Qualitative research shows returning users are hurt by the **caveat line** specifically (not the
  card itself) → test a version without the caveat for returning users before writing them off.
* Real review text yields a different #1 complaint once the tagger is validated on hand-labelled data.
