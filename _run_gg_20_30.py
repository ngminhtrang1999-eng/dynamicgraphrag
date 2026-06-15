"""Run google_trans.py for chapters 20-30 (const overridden at runtime)."""
import const

const.START_CHAPTER = 20
const.END_CHAPTER = 30

exec(open("google_trans.py", encoding="utf-8").read())
