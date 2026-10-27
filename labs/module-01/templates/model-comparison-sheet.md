# Model comparison sheet — {{scenario_id}} {{scenario_title}}

> **Module 1 · Lab 03 artefact.** One sheet per enterprise scenario. Fill it in
> as a team, or let the Lab 03 notebook fill the placeholders in double braces
> and edit the result in `labs/module-01/outputs/`. Prices are list prices on the date shown;
> re-check the provider pricing pages before a real business case.

| Team | Date | Prices as of |
| ---- | ---- | ------------ |
| {{team}} | {{date}} | {{prices_as_of}} |

## 1. Scenario profile

{{scenario_profile}}

## 2. Agent type and model tier

| Decision | Choice | Why |
| -------- | ------ | --- |
| Agent type | {{agent_type}} | {{mapping_rationale}} |
| Model tier | {{model_tier}} | (frontier · balanced · fast) |
| Deployment | {{deployment}} | (public API · regional API · self-hosted open weights) |

## 3. Candidate models

At least one OpenAI and one Anthropic model, plus an open-weight option where
data residency or cost makes it worth considering.

{{candidates_table}}

## 4. Weighted decision matrix

Ratings run from 1 (poor fit) to 5 (excellent fit). Weights sum to 1.0.
Cost is rated from the monthly estimate above: cheapest = 5.

{{matrix_table}}

## 5. Sensitivity check

What would change the winner? (for example: double the cost weight, a
residency rule that forbids the public API, a 3× traffic forecast)

{{sensitivity}}

## 6. Decision

| | |
| --- | --- |
| **Primary model** | {{decision}} |
| **Fallback model** | {{fallback}} |
| **Tie-breaker** (agreed before scoring) | {{tie_breaker}} |
| **Key trade-off** | {{trade_off}} |
| **Review date** | {{review_date}} |

## 7. Responsible AI check (Module 1 §8.5)

| Principle | Question for this scenario | Answer |
| --------- | -------------------------- | ------ |
| Bias | Does quality differ by language, region or user group? How will you measure it? | {{rai_bias}} |
| Transparency | Do users know an AI produced the output, and can they see the sources? | {{rai_transparency}} |
| Governance | Who owns the model choice, the prompts and the eval results? What is logged and for how long? | {{rai_governance}} |

## 8. Risks and open questions

{{risks}}
