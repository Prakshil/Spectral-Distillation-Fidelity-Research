# What we did, in plain language

This is a summary for someone who is not a specialist in this exact area. It
explains what we set out to do, what we found, and why it matters.

## The original goal

We have three real-world fraud-detection graphs. The idea was to build a small
"router" that decides, for each node, which of a few simple prediction models
("experts") should handle it. The hope was that a good router would beat using
one single model for everything.

To tell whether a router is any good, we compared it against a random router and
against an "oracle" router that cheats by peeking at the labels. The quantity we
track is called **D2**: how much better than a random router the router is. A
positive D2 means "this router helps."

## What we discovered along the way

### 1. We were picking candidates in a biased way

The original method first scored every candidate router by how well it
correlated with the true labels, then only tested the top few against those same
labels. That is like choosing which lottery numbers to buy by looking at the
winning numbers. It produced a result that looked real but did not hold up.

We fixed this by testing *all* candidates up front and correcting for the many
comparisons. Result: on the Amazon graph, **none of the 28 label-free routers
we tried was convincingly better than random** once you account for testing 28
of them. The earlier "success" on Amazon was an artifact.

### 2. The sign of a router's score was never doing anything

Every router we build buckets nodes into three groups (e.g. high / medium / low
score) and sends each group to a different expert. We discovered that if you
negate a router's score — flipping high and low groups — the final accuracy is
*exactly the same*. We proved why: each expert is trained only on the group it
is later tested on, so accuracy is just "how good is each group at predicting its
own members," and that does not care which group is called "1" or "3."

This meant a whole step in the original method ("flip the sign to match the
labels") was a no-op. We corrected the write-up that claimed this step was
important.

### 3. The big one: the way we were measuring routers was rigged in their favor

This is the most important finding, and it is the reason we now think carefully
before trusting any router result.

In the original setup, each expert model was trained on exactly the same group of
nodes it was then tested on. Think of it like grading a student on an exam made
of the exact questions you taught them. Of course they do well. And because each
group gets its own expert, *any* way of splitting the nodes into groups — even a
meaningless one — collects some of this "free credit."

So the original measurement could not tell the difference between:
- a router that genuinely routes each node to the model that predicts it best, and
- a router that just chops nodes into arbitrary groups.

To fix this, we changed the test: we built the pool of expert models *first*, in
a way that does not look at the router at all, and *then* varied the routing.
Now the only thing changing between the "good" and "bad" cases is the routing,
so any difference is honestly due to routing.

### 4. Under the honest test, the routers stopped working

When we applied the fair test to the Amazon graph:

- The simple single model (no routing at all) was the best thing we had.
- **None of the 28 label-free routers beat the single model.**
- Even the cheating "oracle" router that was supposed to be the best possible
  label-aware router came out *worse than random*.
- The 28 routers were ranked almost completely differently under the fair test
  than under the old test — which shows the old rankings were measuring the wrong
  thing.

### What this does *not* mean

It does **not** mean routing is useless in general. It means that *on these fraud
graphs, with simple straight-line (linear) experts*, splitting the work up did
not pay for the cost of giving each expert less training data. The honest next
question is whether **stronger, non-linear experts** (like small neural networks)
can specialize well enough for routing to be worth it. That is the experiment we
are running next.

## Why this is worth a serious paper

Three independent checks all point the same way:

1. Correlation with the true labels is a *bad* guide to which router works
   (we saw this three separate times now).
2. The standard way of measuring routers gives them undeserved credit.
3. A common measure of "how similar is a simplified graph to the original"
   silently maxes out at its worst value when the simplified graph falls apart.

The contribution is not "we built a better router." It is **"the yardstick the
field is using to measure routers is broken, here is the proof, here is a correct
way to measure, and here is what happens to the published-style conclusions when
you use it."** That kind of result is valuable to the community and is a
legitimate, well-supported paper — provided we finish the remaining checks
honestly.

## Honest status and what is left

- The Amazon result is solid and reproducible.
- We are re-checking the two other graphs (Tolokers, YelpChi) under the fair test,
  because their earlier "wins" were measured with the unfair test.
- We are testing stronger experts to see whether the negative result is about
  expert strength rather than about routing itself.
- We must also strengthen the work before it is submission-ready: more real
  datasets, a fair comparison against existing published router methods, and
  removing a place where we chose a setting ("k") using the same data we report.