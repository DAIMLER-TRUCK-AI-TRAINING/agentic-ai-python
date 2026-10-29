# Module 3 data

Everything here is **synthetic**, written for training. The company is named because the labs are
tailored for Daimler Truck engineers, but no document, model series, part number, fault code,
specification, policy, contract or procedure is Daimler Truck information. Never use it on a vehicle.

| Path | What it is | Used by |
| ---- | ---------- | ------- |
| `corpus/bulletins/` | 4 technical service bulletins (`TSB-2026-…`), one in German | Labs 01–04 |
| `corpus/workshop/` | 6 workshop repair and fault-code guides (`KB-06…`) | Labs 01–04 |
| `corpus/customer/` | 6 fleet-customer FAQs (`KB-07…`), one in German, `audience: customer` | Labs 01–04 |
| `corpus/it-ops/` | 4 dealer and plant IT runbooks (`KB-08…`): the Module 2 IT-operations thread | Labs 01–04 |
| `connectors/sharepoint/` | Export of the "WarrantyCompliance" library: 8 files plus `manifest.json` with the library columns and permissions, including one **retired** policy version | Lab 03 |
| `connectors/confluence/pages.json` | Export of the "ENGDOCS" space in Confluence REST shape (storage-format XHTML, page restrictions) | Lab 03 |
| `test_queries.json` | 27 labelled queries: identifiers, paraphrases, German and cross-lingual, 3 unanswerable | Lab 02 |

## The fictional product world

| Name | What it is |
| ---- | ---------- |
| LH-40 | Long-haul diesel tractor with the DTE-13 engine |
| ED-26 | Battery-electric distribution truck with the eAxle EAX-2 |
| CT-33 | Construction tipper with the DTE-11 engine and the AC-2 air compressor |
| TCU-5 | Telematics control unit; over-the-air updates through the Fleet Insight portal |
| DiagLink, VCI-3 | Workshop diagnostic software and its vehicle interface |
| Dealer Workshop Portal | Dealers' portal for warranty claims |

## Metadata schema

Every record, whatever its source, has the same fields: `doc_id`, `title`, `doc_type` (`bulletin`,
`kb`, `faq`, `runbook`, `policy`, `contract`, `engineering`), `product`, `owner_team`, `region`
(`EU`, `US`, `ALL`), `audience` (`internal`, `dealer`, `customer`), `language`, `updated`,
`allowed_groups`, `source`, and for connector documents `status`, `version` and `url`.
`rag_kit.flat_metadata()` turns `allowed_groups` into one `acl_<group>=1` key per group for the
vector stores.
