from datetime import datetime, timezone, timedelta


def get_now_jst() -> datetime:
    now_utc = datetime.now(timezone.utc)
    now_jst = now_utc.astimezone(timezone(timedelta(hours=9)))

    return now_jst
