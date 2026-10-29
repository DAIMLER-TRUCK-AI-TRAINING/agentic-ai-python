# Telematics and vehicle data residency

## §1 Scope

This policy applies to vehicle data from connected trucks (positions, fault codes, battery and
engine data) and to every copy, index or cache derived from it.

## §2 Residency by region

| Vehicle registered in | Data stored and processed in |
| --------------------- | ---------------------------- |
| EU or EEA | EU cloud regions only |
| United States | US cloud regions only |
| Asia-Pacific | The APAC region of the vehicle's country |

## §3 Derived data

Search indexes, vector indexes, caches and analytics extracts follow the residency of the data
they were built from. A vector index built from EU documents or EU vehicle data must be hosted
in an EU region, and it must be rebuilt or purged when a source record is deleted.

## §4 Transfers

Data may leave its region only after anonymisation approved by the data protection officer,
and only for aggregated analytics.

## §5 Retention

Raw position data is deleted after 90 days unless a customer contract requires longer.
