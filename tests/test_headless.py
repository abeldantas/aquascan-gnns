from aquascan.run_simulation import run

def test_5_ticks_headless():
    run(ticks=5, visual=False, seed=123)
