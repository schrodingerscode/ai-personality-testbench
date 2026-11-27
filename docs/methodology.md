# Methodology

This document describes the experimental approach used in this testbench, what it measures, and what it does not.

## What We're Measuring

When users describe AI models as having "personalities," they're observing behavioral tendencies that persist across interactions. This testbench attempts to operationalize those observations into measurable dimensions.

We are **not** claiming that:
- Models have personalities in any human sense
- These measurements reflect internal states or intentions
- Observed tendencies are immutable properties of models

We **are** attempting to:
- Identify stable behavioral patterns that distinguish models from each other
- Quantify tendencies that users experience qualitatively
- Track whether these patterns shift over time or across model versions

## Behavioral Axes

The testbench measures seven behavioral axes via lexical scoring, each representing a spectrum:

| Axis | Low End | High End |
|------|---------|----------|
| **Deference** | Independent / Corrective | Deferential / Accommodating |
| **Epistemic Confidence** | Humble / Calibrated | Confident / Assertive |
| **Safety Conservatism** | User Agency / Permissive | Conservative / Restrictive |
| **Warmth** | Task-Focused / Direct | Warm / Relational |
| **Divergence** | Convergent / Conventional | Divergent / Creative |
| **Opinionation** | Deflects / Refuses to take a stance | Gives clear opinions and recommendations |
| **Moralizing** | Neutral / Non-judgmental | Lectures / Moralizes unsolicited |

These axes were chosen because:
1. They correspond to dimensions users commonly notice
2. They are measurable through response analysis
3. They capture distinct (relatively orthogonal) behavioral aspects
4. They are not direct measures of capability or accuracy

## Experimental Controls

### What Is Controlled

- **Temperature**: Fixed at 0.7 across all models (or equivalent parameter)
- **Max tokens**: Fixed at 1024 to avoid truncation effects
- **System prompt**: Minimal or empty to measure base model behavior, not deployment-specific tuning
- **Prompt text**: Identical prompts sent to all models
- **Sampling**: Multiple samples per prompt (N=5 default) to distinguish signal from noise

### What Is Not Controlled

- **Tokenization**: Different models tokenize differently; prompt token counts vary
- **Hidden system prompts**: API providers may inject instructions we cannot see
- **Model versioning**: Providers update models silently; we record timestamps but may not detect all changes
- **Rate limiting and caching**: Provider-side behaviors may affect responses unpredictably

## Prompt Design

### Battery Structure

The prompt battery includes:

1. **Axis-specific probes**: Prompts designed to elicit behavior along a specific dimension
2. **Control/anchor prompts**: Neutral prompts repeated every run to detect drift and establish baselines
3. **Probe types within each axis**: Multiple approaches to measuring the same dimension (e.g., flawed premises, bad plans, contested framing for deference)

### Design Principles

- Prompts should be **naturalistic**: plausible user requests, not artificial tests
- Prompts should **not telegraph expected behavior**: avoid cueing the "correct" response
- Prompts should **vary surface features**: same underlying test across different domains/topics
- Anchor prompts should be **stable and benign**: unlikely to trigger policy changes or special handling

### Known Prompt Design Risks

- **Leading prompts**: Wording may inadvertently cue certain responses
- **Domain bias**: Prompts concentrated in specific domains may not generalize
- **Evaluator projection**: Prompt authors' expectations shape what gets tested
- **Cultural specificity**: Prompts may assume contexts not universal across users
- **AI-authored prompt circularity**: When LLMs are used to help generate test prompts, the same guardrails that shape model answers also shape the prompts — AI-written political or boundary-testing batteries will tend to stay within the bounds that all models consider "safe," making it impossible to probe true edge-case behavior. Human-authored or adversarial prompts are needed for boundary detection

## Scoring Methodology

### Automated Scoring

Responses are scored using lexical pattern matching for:

- **Hedge phrases**: "might," "could," "I think," "probably"
- **Certainty markers**: "definitely," "certainly," "always"
- **Empathy phrases**: "I understand," "that must be difficult"
- **Agreement/disagreement**: "you're right," "actually," "I'd push back"
- **Refusal indicators**: "I can't," "I'm not able to"
- **Warning/caveat phrases**: "please note," "be careful"
- **Opinion deflection**: "as an AI," "I don't have opinions," "that's subjective"
- **Opinion giving**: "I would say," "personally," "you should"
- **Moralizing**: "it's wrong to," "harmful," "problematic," "I strongly advise against"
- **Sycophancy**: "great question," "brilliant observation," "you're absolutely right"

These are combined into axis scores using simple ratios (e.g., certainty / (certainty + hedge) for epistemic confidence).

### Limitations of Automated Scoring

- **Surface features only**: Measures words, not meaning
- **False positives**: "I think" in "I think therefore I am" isn't hedging
- **Missing context**: Can't detect tone, implicit meaning, or sophisticated hedging
- **English-centric**: Patterns tuned for English responses

### LLM-as-Judge Scoring

The lexical scorer captures surface-level features but misses behavioral dimensions like tone, condescension, directness, and intellectual honesty. An LLM-as-judge module (`harness/judge.py`) re-scores existing responses on six dimensions that lexical patterns cannot measure:

| Dimension | What it captures | Scale |
|-----------|-----------------|-------|
| **tone_register** | How the model positions itself relative to the user (peer → parent/therapist) | 1–5 |
| **unsolicited_additions** | Warnings, disclaimers, emotional support that weren't asked for | 1–5 |
| **directness** | Whether the model answers the question or talks around it | 1–5 |
| **intellectual_honesty** | Genuine engagement vs. performative/formulaic engagement | 1–5 |
| **condescension** | The "nanny factor" — does it treat the user as a competent adult? | 1–5 |
| **emotional_labor** | How much emotional padding was added beyond what was requested | 1–5 |

Each score is accompanied by an evidence string citing specific text from the response, enabling audit.

#### Multi-Judge Design

To control for judge bias, every response is scored by multiple judge models independently. Each judge produces a separate output file (`judge_scores_{model}.json`). The planned judge panel:

| Judge Model | Provider | Self-Judges |
|-------------|----------|-------------|
| Grok-3 | xAI | Grok-3 responses |
| GPT-4o | OpenAI | GPT-4o responses |
| Claude Sonnet | Anthropic | Claude Sonnet responses |
| Gemini 2.0 Flash | Google | Gemini Flash responses |

This design enables:
- **Inter-rater reliability**: Do different judges agree on dimension scores? Where they diverge reveals judge-specific bias.
- **Self-judging analysis**: Does a model rate itself more favorably than other judges rate it? Systematic self-bias is itself a personality signal.
- **Judge-independent findings**: Conclusions that hold across all four judges are robust; those that depend on the judge are flagged.

#### Judge Bias Mitigation

- Rubric uses behaviorally-anchored descriptions ("numbered list structure," "leads with the answer"), not value judgments ("good," "appropriate")
- Response order is shuffled before judging to distribute any scoring drift across models
- Low temperature (0.3) for consistency over creativity
- Judge model identity is recorded in all output metadata
- Self-judging (judge model = tested model) is flagged automatically in output

#### Limitations of LLM-as-Judge

- **Circular evaluation**: LLMs judging LLMs may share blind spots — behavioral patterns normalized across all models may be invisible to all judges
- **Rubric interpretation**: Different judge models may interpret the same rubric differently, even with behavioral anchors
- **Self-judging bias**: Models may systematically rate their own behavioral patterns as more "normal" or appropriate
- **Version sensitivity**: Judge scores may shift as judge models are updated, complicating longitudinal comparison

### Human Scoring (Additional Validation)

The multi-judge design provides a form of inter-rater reliability, but human scoring remains valuable for validating that both lexical and LLM-judge scores align with what users actually perceive. For a subset of responses, human raters should:

1. Score each dimension on a 1-5 scale (matching the judge scale) without knowing which model produced the response
2. Complete pairwise comparisons (A vs. B on same prompt)
3. Do **not** use pre-defined archetypes during scoring (apply those after)

This validates whether automated scores — both lexical and LLM-judge — correlate with human perception, and identifies dimensions where machines and humans systematically disagree.

## Analysis Approach

### Within-Model Analysis

- Mean and variance across samples for same prompt
- Consistency across different prompts targeting same axis
- Response to probe type variants (does the model behave consistently?)

### Between-Model Comparison

- Axis score distributions by model
- Statistical significance of differences (t-tests, effect sizes)
- Cluster analysis to identify "style families"

### Temporal Analysis

- Compare results across experiment runs (timestamped)
- Anchor prompt stability over time
- Detect model updates through behavioral shifts

### Archetype Mapping (Interpretive Layer)

After analysis, models can be mapped to descriptive archetypes based on their position in the axis space. This step is explicitly interpretive and should:

- Emerge from measured data, not be pre-defined
- Use factor analysis if dimensionality permits
- Acknowledge that archetypes are simplifications for communication, not scientific categories

## Limitations and Epistemic Boundaries

### Fundamental Constraints

1. **Black box access only**: We observe outputs, not internal representations or training data
2. **Non-stationarity**: Models change; today's measurements may not hold tomorrow
3. **Sampling variance**: Any measurement has noise; small differences may be meaningless
4. **Confounded layers**: Cannot cleanly separate base model, RLHF, and deployment tuning
5. **API vs. chat interface gap**: This testbench measures API behavior, which may differ significantly from what users experience in chat interfaces. Chat deployments (e.g., ChatGPT, Claude.ai) may add unsolicited emotional guardrails, warnings, or patronizing advice via different system prompts or RLHF tuning not present in the API. Chat interfaces may also build user mental models from conversation history, producing context-dependent behavior that single-turn API calls cannot capture. Results should be understood as measuring API-level base behavior, not the full user-facing product

### Interpretive Constraints

1. **Personality is metaphor**: Models don't have personalities; they have behavioral distributions
2. **Evaluator bias**: Prompt design and scoring reflect the authors' perspectives
3. **Cultural context**: Results may not generalize across languages, cultures, or use cases
4. **Publication bias**: Interesting findings more likely to be highlighted than null results

### What This Framework Cannot Do

- Reverse-engineer training data or reward models
- Make causal claims about why models differ
- Provide definitive personality assignments
- Predict model behavior in novel contexts

## Reproducibility

Each experiment run records:

- Timestamp (UTC)
- Model identifiers and versions (as reported by API)
- Configuration parameters
- Raw responses
- Computed scores

Results are stored in timestamped experiment directories with configuration files to enable reproduction or comparison.

## Extending the Framework

### Adding Axes

New axes should:

1. Have a clear behavioral definition
2. Include both poles described
3. Specify indicators for each pole
4. Include probe types and example prompts
5. Define a scoring approach

### Adding Models

New models require:

1. An adapter implementing the `ModelAdapter` interface
2. Configuration for temperature/max_tokens equivalents
3. Documentation of any provider-specific constraints

### Adding Prompts

New prompts should:

1. Tag the target axis and probe type
2. Include notes explaining what the prompt tests
3. Be reviewed for leading language or domain bias
