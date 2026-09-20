# ═══════════════════════════════════════════════════════════════════════
#  Reading earnings releases, in production.
#
#  Research built this on stored filings (research/v3_*.py); this package
#  is the same pipeline in a form the daily run can use on a filing that
#  appeared an hour ago:
#
#    parse     EDGAR exhibit HTML -> paragraphs and table rows
#    novelty   which sentences are new, against the company's own past
#    readers   the three models, loaded once, scoring sentences
#    features  reader output -> the features the fitted model expects
#    model     the fitted model itself (config/text_model.npz)
#    fetch     today's 8-K from EDGAR, rate-limited
#    store     the filing cache and the shadow record
#
#  Nothing here decides a real trade. The text signal runs in shadow mode
#  until it passes the gate in CLAUDE.md.
# ═══════════════════════════════════════════════════════════════════════
