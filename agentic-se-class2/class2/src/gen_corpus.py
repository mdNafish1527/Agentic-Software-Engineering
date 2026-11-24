"""Builds a deterministic 50-notice corpus + answer key (written before any model run).
Replace corpus/notices.jsonl and corpus/answer_key.jsonl if the class issues its own corpus."""
import json, random, datetime, pathlib

rng = random.Random(2026)
ROOT = pathlib.Path(__file__).resolve().parent.parent
REGS = ["NetNimbus Registrar", "Padma Domains Ltd", "BlueHarbor Names", "Jamuna Hosting & Domains",
        "OrbitDNS Inc.", "Karnaphuli Web Services", "NorthStar Registry", "Meghna Online Ltd"]
WORDS = ["alpha", "riverbank", "greenleaf", "shopnoghor", "techvilla", "bongo", "urbanfit", "cloudnine",
         "dhakabytes", "sunrise", "kitebuilders", "lotusmart", "pixelforge", "nilkhet", "tarapress"]
TLDS = [".com", ".net", ".org", ".com.bd", ".io", ".xyz"]
CURS = ["USD", "BDT", "EUR", "GBP"]
TYPES = ["expiry_reminder", "renewal_confirmation", "transfer_request",
         "suspension_notice", "dispute_notice", "verification_request"]


def fmt_date(d):
    return rng.choice([d.isoformat(), d.strftime("%d %B %Y"), d.strftime("%B %d, %Y")])


def money(a, c):
    return rng.choice([f"{c} {a:,.2f}", f"{a:,.2f} {c}"])


def make(i):
    t = TYPES[i % 6] if i < 48 else rng.choice(TYPES)
    dom = rng.choice(WORDS) + str(rng.randint(1, 99)) + rng.choice(TLDS)
    reg = rng.choice(REGS)
    issue = datetime.date(2026, 1, 1) + datetime.timedelta(days=rng.randint(0, 300))
    dl = issue + datetime.timedelta(days=rng.randint(7, 45))
    cur = rng.choice(CURS)
    amt = round(rng.uniform(800, 4500), 2) if cur == "BDT" else round(rng.uniform(8, 120), 2)
    I, D = fmt_date(issue), fmt_date(dl)
    head = f"From: {reg}\nDate: {I}\n"
    key = dict(domain=dom, notice_type=t, registrar=reg, issue_date=issue.isoformat(),
               deadline_date=None, amount=None, currency=None, action_required=True)
    if t == "expiry_reminder":
        body = rng.choice([
            f"Subject: {dom} is about to expire\n\nThe registration of {dom} ends on {D}. Renew before then to keep the domain active. Renewal fee: {money(amt, cur)}.",
            f"Subject: Renewal reminder\n\nDear customer, your domain {dom} will expire on {D}. Please renew it at {money(amt, cur)} to avoid losing it."])
        key.update(deadline_date=dl.isoformat(), amount=amt, currency=cur)
    elif t == "renewal_confirmation":
        body = rng.choice([
            f"Subject: Renewal successful\n\nWe have renewed {dom}. Payment received: {money(amt, cur)}. Nothing more is needed from you.",
            f"Subject: Receipt for {dom}\n\nThank you. {dom} is renewed and {money(amt, cur)} was charged. No action required."])
        key.update(amount=amt, currency=cur, action_required=False)
    elif t == "transfer_request":
        body = rng.choice([
            f"Subject: Transfer request for {dom}\n\nA request to transfer {dom} to another registrar was received. Approve or reject it by {D}, otherwise it proceeds automatically.",
            f"Subject: Outbound transfer pending\n\nWe received an outbound transfer order for {dom}. If you did not request it, reject it before {D}."])
        key.update(deadline_date=dl.isoformat())
    elif t == "suspension_notice":
        body = rng.choice([
            f"Subject: {dom} suspended\n\nThe domain {dom} has been suspended because the contact details are inaccurate. Update them by {D} or the domain will be deleted.",
            f"Subject: Suspension notice\n\n{dom} is suspended pending WHOIS correction. Submit valid registrant information no later than {D}."])
        key.update(deadline_date=dl.isoformat())
    elif t == "dispute_notice":
        body = rng.choice([
            f"Subject: Domain dispute filed against {dom}\n\nA UDRP complaint has been filed concerning {dom}. A written response is due by {D}.",
            f"Subject: Complaint regarding {dom}\n\nA third party has lodged a dispute over {dom}. You must reply by {D} to contest it."])
        key.update(deadline_date=dl.isoformat())
    else:
        body = rng.choice([
            f"Subject: Verify your email for {dom}\n\nPlease confirm the registrant email address for {dom} by {D}, or the domain may be suspended.",
            f"Subject: Action needed: contact verification\n\nWe need you to verify the contact email of {dom} before {D}."])
        key.update(deadline_date=dl.isoformat())
    nid = f"N{i+1:03d}"
    return {"id": nid, "text": head + "\n" + body}, {"id": nid, "fields": key}


pairs = [make(i) for i in range(50)]
(ROOT / "corpus/notices.jsonl").write_text("\n".join(json.dumps(n, ensure_ascii=False) for n, _ in pairs) + "\n", encoding="utf-8")
(ROOT / "corpus/answer_key.jsonl").write_text("\n".join(json.dumps(k, ensure_ascii=False) for _, k in pairs) + "\n", encoding="utf-8")
print("wrote 50 notices + answer key")
