# Fatty-acid chain fixtures — hand-computable ground truth

Toy datasets for evaluating chain-level fatty-acid questions (e.g. *"what is the
C18/C16 ratio per sample?"*), motivated by a real user question about an **ELOVL6
knockout** — ELOVL6 elongates C16 (palmitate) to C18, so a KO is expected to show a
**lower C18/C16 ratio** than control.

## Why these exist

The C18/C16 computation is not a built-in LipidCruncher feature and is not trivial to
do by hand on a real dataset. That creates an *oracle problem*: a domain expert can
judge the **method** but cannot produce the **number** independently.

These fixtures resolve that. They are deliberately tiny, so the expected answer is
computable with a calculator and needs no tool to verify. The logic under test is
identical to the real case; only the scale is reduced.

**Ground truth here is arithmetic, not opinion.** What still needs an expert decision
is the *counting convention* — see [Decisions required](#decisions-required).

## Conventions under test

Two defensible ways to pool intensity by carbon number (the AI is prompted to ask
which one the user wants — see `src/app/services/ai_chat/prompt_templates.py`):

- **Presence-based** — a species contributes its **full** intensity to every pool it
  touches. `PC(16:0_18:1)` at 200 adds 200 to the C16 pool *and* 200 to the C18 pool.
- **Chain-split** — a species' intensity is **divided equally across its chains**.
  `PC(16:0_18:1)` at 200 adds 100 to C16 and 100 to C18. (Mass-conserving.)

Neither is "right" — it depends on the biological question. Both expected values are
given below so the fixture stays convention-agnostic.

---

## Fixture 1 — `chain_molecular.csv` (the answerable case)

Molecular form (individual chains resolvable). `s1,s2` = control; `s3,s4` = ELOVL6 KO.

| Lipid | chains | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|
| `PC(16:0_16:0)` | C16, C16 | 100 | 200 | 300 | 150 |
| `PC(16:0_18:1)` | C16, C18 | 200 | 400 | 200 | 100 |
| `PC(18:0_18:1)` | C18, C18 | 300 | 600 | 100 | 50 |

### Expected values

| Sample | Convention | C16 pool | C18 pool | **C18/C16** |
|---|---|---|---|---|
| s1 (control) | presence-based | 300 | 500 | **1.667** |
| s1 (control) | chain-split | 200 | 400 | **2.000** |
| s2 (control) | presence-based | 600 | 1000 | **1.667** |
| s2 (control) | chain-split | 400 | 800 | **2.000** |
| s3 (KO) | presence-based | 500 | 300 | **0.600** |
| s3 (KO) | chain-split | 400 | 200 | **0.500** |
| s4 (KO) | presence-based | 250 | 150 | **0.600** |
| s4 (KO) | chain-split | 200 | 100 | **0.500** |

<details>
<summary>Worked example — s1, both conventions</summary>

**Presence-based (s1: 100, 200, 300)**
- C16 pool = species containing ≥1 C16 chain = `PC(16:0_16:0)` 100 + `PC(16:0_18:1)` 200 = **300**
- C18 pool = species containing ≥1 C18 chain = `PC(16:0_18:1)` 200 + `PC(18:0_18:1)` 300 = **500**
- ratio = 500 / 300 = **1.667**

**Chain-split (s1: 100, 200, 300)**
- `PC(16:0_16:0)` 100 → both chains C16 → C16 += 100
- `PC(16:0_18:1)` 200 → split → C16 += 100, C18 += 100
- `PC(18:0_18:1)` 300 → both chains C18 → C18 += 300
- C16 = **200**, C18 = **400** → ratio = **2.000**
</details>

### Properties this fixture pins

1. **Biological direction (positive control).** KO < control under **both** conventions
   (0.600 vs 1.667; 0.500 vs 2.000). This is the assertion the biologist can make
   independently, without computing anything. A pipeline reporting KO > control is wrong.
2. **Scale invariance.** `s2 = 2 × s1` and `s4 = 0.5 × s3`, so the ratio must be
   unchanged within each condition. Catches normalization/scaling bugs for free.
3. **Convention discrimination.** The two conventions give *different* answers
   (1.667 vs 2.000), so the fixture reveals which one an implementation actually used.
4. **Mass balance (chain-split only).** C16 + C18 pools must equal total intensity
   exactly: s1 → 200 + 400 = 600 = 100 + 200 + 300. ✔

---

## Fixture 2 — `chain_consolidated.csv` (the must-refuse case)

**Identical intensities to Fixture 1**, but with sum-composition names — the individual
chains are *not* recoverable:

| Lipid | s1 | s2 | s3 | s4 |
|---|---|---|---|---|
| `PC(32:0)` | 100 | 200 | 300 | 150 |
| `PC(34:1)` | 200 | 400 | 200 | 100 |
| `PC(36:1)` | 300 | 600 | 100 | 50 |

`PC(34:1)` is ambiguous: it could be `16:0_18:1`, `16:1_18:0`, or `17:0_17:1`. A
C18/C16 ratio computed from this data is **numerically producible but biologically
meaningless**.

### Expected behaviour

> **The correct answer is a refusal, not a number.** The assistant must detect the
> consolidated form and state that individual C16/C18 chains cannot be resolved from
> sum-composition data — rather than confidently returning a ratio.

This is the epistemic-honesty case. Because the intensities are identical to Fixture 1,
the pair forms a clean A/B: **same numbers, one answerable, one not.** Any pipeline that
returns the same ratio for both is failing.

---

## Fixture 3 — `chain_edge_cases.csv` (pins the sphingoid & ether rules)

Two samples, all rows at 100.

| Lipid | chain inventory |
|---|---|
| `PC(16:0_18:1)` | `16:0` acyl (C16), `18:1` acyl (C18) |
| `Cer(d18:1/16:0)` | `d18:1` **sphingoid base** (C18?), `16:0` acyl (C16) |
| `PC(O-16:0_18:1)` | `O-16:0` **ether chain** (C16?), `18:1` acyl (C18) |

### Expected values (presence-based), per sample

| # | Sphingoid base | Ether chain | C16 pool | C18 pool | C18/C16 |
|---|---|---|---|---|---|
| a | counts as C18 | counts as C16 | 300 | 300 | 1.00 |
| b | **excluded** | counts as C16 | 300 | 200 | 0.67 |
| c | counts as C18 | **excluded** | 200 | 300 | 1.50 |
| d | **excluded** | **excluded** | 200 | 200 | 1.00 |

Note rows **a** and **d** share a ratio of 1.00 but differ in the **pools** — so compare
pools, not just the ratio, to identify which rule an implementation used.

---

## Decisions required

These are **biology decisions, not arithmetic** — the fixtures deliberately leave them
open. A domain expert must choose, and the choice then becomes the expected value used
by the eval harness:

- [ ] **Counting convention** — presence-based or chain-split? (Fixture 1)
- [ ] **Sphingoid bases** — does `d18:1` in `Cer(d18:1/16:0)` count as a C18 fatty acid,
      or is the base excluded from FA pools? (Fixture 3)
- [ ] **Ether/plasmalogen chains** — does `O-16:0` count as a C16 fatty acid, or is the
      ether chain excluded? (Fixture 3)
- [ ] **Consolidated data** — confirm that refusal (not a number) is the correct
      behaviour for Fixture 2.

### Sign-off

| Field | Value |
|---|---|
| Reviewed by | _(name)_ |
| Date | _(YYYY-MM-DD)_ |
| Convention chosen | _(presence-based / chain-split)_ |
| Sphingoid rule | _(count as C18 / exclude)_ |
| Ether rule | _(count as C16 / exclude)_ |

---

## Scope notes

- **Format.** These are **Generic** format (`Lipids` + `sample[sN]` columns). Chain
  parsing happens after standardization, so the chain logic is format-independent —
  covering other input formats is a separate axis (format detection/ingestion), not a
  chain-arithmetic concern.
- **These fixtures test logic, not scale.** They give *exact* ground truth. Confidence on
  real datasets comes from different tools: invariants (mass balance, scale invariance),
  the known biological direction, and an independent reference implementation.
- **Not a substitute for expert review of method.** The fixtures verify arithmetic. Whether
  the *approach* is biologically sound remains an expert judgement.
