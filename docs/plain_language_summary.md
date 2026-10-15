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

### 5. We then asked: was the negative just because our experts were too weak?

This is the obvious objection, and a negative result is worthless if the experts
are too simple to do anything interesting. So we ran the whole thing again with
much stronger experts:

- a **neural network** expert (instead of a straight-line one), and
- a **graph neural network** expert, which is a model that actually looks at each
  node's neighbours. This is the kind of expert used in the published routing
  work, so it is the fairest test of the idea.

First we checked the new experts were actually good, not broken. On a small
test problem whose answer depends only on a node's neighbours, ordinary experts
score *no better than a coin flip* (0.52), while the graph expert reaches 0.93 —
essentially as good as knowing the answer in advance (0.94). So the graph expert
genuinely does understand the structure it is given.

Then we re-ran everything. Across **all 234 combinations** of router, expert type,
and graph:

**None of the 234 beats simply training one model on everything — every cell is
0 of 26. An earlier revision reported two tiny exceptions on a reduced-power
YelpChi GNN arm (positive by ~0.002, p >= 0.75, i.e. noise); at full power even
those disappear.**

The graph expert is the interesting case, because it is the strongest single
demonstration that the structure is exploitable — and it still changes nothing.
No candidate beats the single model, and once we correct for testing 26
candidates at once, not one of them beats a random split either (0 of 26, versus
16 of 26 before the correction). So:

- Beating *random* routing was the wrong bar, and the graph expert does not even
  clear that once the comparison is corrected.
- The real question is whether routing beats *no routing*, and it never does.

### A bug we found along the way

While setting up the graph expert we discovered our own "no routing" baseline was
not doing what its name said. Because the test reuses a fixed pool of three
experts, the "single model" was actually being judged by one of those three
experts — which had only seen a third of the training data. That made the
baseline too *weak*, which would have flattered our routers.

We measured the size of the mistake (under 0.5%, and in the direction that made
us look worse, not better), fixed it, regenerated every result, and added a test
so it cannot come back. The conclusion did not change.

### Two more checks, so the negative result is not just us being wrong

A negative result is only worth something if the machine could have produced a
positive one. So we added two controls.

**A test problem where routing must win.** We built a synthetic graph whose answer
is three different linear rules in three different regions — a shape no single
straight line can fit, but three local experts can. On that problem, knowing which
expert to send a node to raises accuracy from 0.652 to 0.988. Splitting the work is
worth +0.34 there. Random splitting does *not* help (0.627), so this is not a
harness that flatters every option. It also shows why routing looked unpromising:
with a flexible neural expert the gain shrinks to +0.01, because one big model can
already do the job. Routing only pays when the global model is too simple.

**A router that learns from the labels.** Every router above was a fixed formula.
We also trained a small graph neural network to pick the expert per node, using
only training labels, against the same frozen pool. It found +0.31 of the available
+0.33 on the synthetic problem — so the trained router works. On the three real
fraud graphs it landed within 0.002 of just training one model, and the sign
flipped from graph to graph: noise, not signal.

Taken together: the check that could have produced a win did produce one, and the
three real graphs still produced none.

### The learnability lever: when *should* routing win?

The synthetic test above is all-or-nothing: either routing wins or it does not.
But "routing never wins on these graphs" is only meaningful if we know *where*
routing would start to win. So we built a single dial that slides a task from
"routing is impossible" to "routing is trivial," keeping the graph and the data
shape fixed.

At the zero end, every region obeys the *same* rule, so one model for everything
is perfect and routing can add nothing. At the top end, each region obeys its own
rule — the synthetic case above. Turning the dial in between, we measured both
whether routing *could* help (the cheating oracle) and whether a real router
*actually* did.

What we found:

- Routing first becomes **possible** when the dial reaches about **0.2**. Below
  that, even the cheating oracle gains nothing.
- Routing becomes **reliably worth doing** a little higher, around **0.3**.
  Below that, a router trained on labels is too uncertain to commit.
- On the two real fraud graphs, a router that is allowed to *decline* correctly
  declines: its held-out check says routing would not help, and it sends
  everything to the single global model. That is the same null result as before,
  but now it is a decision the router reached, not a failure to find a win.

So the negative result is not "our router could not find a win anywhere." It is
"there is a threshold of how region-specific the rules must be for routing to pay
off, the real fraud graphs sit below it, and a router that can measure this
notices and stands down."

A side observation: making the graph *less* clustered (mixing edges across
regions) barely changed the routing gain. In this setup the routing signal comes
from the *label/feature* structure, not from which nodes are connected — so
"these graphs are too heterophilous" is not the reason routing fails; the regions
just are not different enough.

### What this does *not* mean

It does **not** mean routing is useless in general. It means that *on these fraud
graphs*, splitting the work up does not pay for the cost of giving each expert
less training data — and this holds even when the experts are neural networks that
can see structure, not just straight lines, and even when the router is trained on
the labels.

Two limits are worth being explicit about. We have not re-implemented the
published routing methods themselves (Ada-Routing and similar), so this is a result
about *structural-score* and *our own* trained router rather than a direct
head-to-head with prior work.

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

Done and reproducible:

- The fair protocol, with the weak-baseline bug found and fixed, and a test that
  stops it returning.
- All three graphs (Amazon, Tolokers, YelpChi) × three expert families × 26
  routers = 234 comparisons. None beats one model; every cell is 0 of 26 at full
  power (an earlier reduced-power YelpChi GNN arm showed two nominal exceptions,
  both at p >= 0.75).
- A positive control where routing wins by +0.34, so the null result is not a
  broken harness.
- A label-trained router that recovers +0.31 of +0.33 on the positive control and
  nothing on the real graphs.
- A **learnability lever** (`docs/learnability_lever.md`) that locates where
  routing starts to pay off: it becomes *possible* at β ≈ 0.2 and *reliably worth
  doing* by β ≈ 0.5, and a selective router that abstains below the threshold
  correctly declines on the real fraud pools (held-out gain +0.0007 on Amazon,
  −0.0017 on Tolokers).

Still to do before submission:

- ~~A fair head-to-head against the published learned routers (RouterGNN,
  Ada-Routing/AD-GNN)~~ — **done** (`src/published_routers.py`): both have
  negative mean deltas on all six graphs except YelpChi (RouterGNN +0.0015, not significant; win 4/18 and 1/18 runs at the ±0.0005
  noise scale).
- More real datasets.
- Removing a place where we chose a setting ("k") using the same data we report.