"""Mutation check: every mutant of om/dnn.om must fail tests/test_ops.py.

Usage: python tests/mutants.py [cases]   (OMEGA_MV / OMEGA_BRIDGE as for tools/build.sh)
Each mutant replaces exact lines of the unit, is rebuilt into its own directory and run against the
randomized suite; a mutant that builds and passes is a SURVIVED row and makes the exit status 1.
A mutation whose text is not found aborts (the list is stale).
"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = os.path.join(ROOT, "om", "dnn.om")

MUTANTS = [
    ("silu sign", [("ret div v add 1.0f64 fexp neg v", "ret div v add 1.0f64 fexp v")]),
    ("rms mean divides by n-1", [("s = fsqrt add div s f64 get p 5 pf p 6", "s = fsqrt add div s f64 sub get p 5 1 pf p 6")]),
    ("gelu tanh cubic coefficient", [("u = mul 0.7978845608028654f64 add v mul 0.044715f64 mul v mul v v",
                                      "u = mul 0.7978845608028654f64 add v mul 0.04471f64 mul v mul v v")]),
    ("layernorm drops eps", [("q = fsqrt add div q f64 get p 5 pf p 6", "q = fsqrt div q f64 get p 5")]),
    ("softmax without max subtraction", [
        ("s = add s fexp sub ldfo ws add b c m", "s = add s fexp ldfo ws add b c"),
        ("v = div fexp sub ldfo ws add b c m s", "v = div fexp ldfo ws add b c s"),
    ]),
    ("rope second half sign", [("y = add mul x2 cs mul x1 sn", "y = sub mul x2 cs mul x1 sn")]),
    ("causal mask excludes the diagonal", [("put p 30 minu add rem r get p 8 1 get p 9", "put p 30 minu rem r get p 8 get p 9")]),
    ("GQA head mapping", [("hk = div rem div r get p 8 get p 6 div get p 6 get p 7", "hk = rem rem div r get p 8 get p 6 get p 7")]),
    ("paged decode ignores sequence length", [("put p 27 minu sub get p 31 mul pg get p 11 get p 11", "put p 27 get p 11")]),
    ("conv bias channel", [("s = add s ldf ws add get p 4 rem get p 30 get p 10", "s = add s ldf ws get p 4")]),
    ("im2col block column offset", [("rb = add get p 5 add mul r get p 29 get p 13", "rb = add get p 5 mul r get p 29")]),
    ("maxpool starts at zero", [("put p 29 bitcast ulong -1.0e300f64", "put p 29 bitcast ulong 0.0f64")]),
    ("avgpool count_include_pad area", [("put p 30 mul get p 16 get p 17", "put p 30 add get p 16 get p 17")]),
    ("fusion never virtualises", [("if ne and arg d i 15 2 0", "if eq i i")]),
]


def main():
    cases = sys.argv[1] if len(sys.argv) > 1 else "20"
    text = open(SRC).read()
    survived = 0
    py = sys.executable
    for name, subs in MUTANTS:
        mtext = text
        for old, new in subs:
            if mtext.count(old + "\n") != 1:
                sys.exit(f"mutants.py: '{old}' is not one exact line of om/dnn.om")
            mtext = mtext.replace(old + "\n", new + "\n")
        with tempfile.TemporaryDirectory(prefix="omega-dnn-mut-") as tmp:
            src = os.path.join(tmp, "dnn.om")
            open(src, "w").write(mtext)
            env = dict(os.environ, OMEGA_DNN_SRC=src, OMEGA_DNN_OUT=os.path.join(tmp, "build"))
            b = subprocess.run(["sh", os.path.join(ROOT, "tools", "build.sh")], env=env, capture_output=True, text=True)
            if b.returncode != 0:
                print(f"KILLED(build)  {name}: rc={b.returncode} {(b.stderr or b.stdout).strip().splitlines()[-1:]}")
                continue
            env["OMEGA_DNN_BUILD"] = env["OMEGA_DNN_OUT"]
            t = subprocess.run([py, os.path.join(ROOT, "tests", "test_ops.py"), cases], env=env, capture_output=True, text=True)
            hit = [ln.split()[0] for ln in t.stdout.splitlines() if "violations=" in ln and not ln.endswith("violations=0") and not ln.startswith("TOTAL")]
            if t.returncode == 0:
                survived += 1
                print(f"SURVIVED       {name}")
            elif hit:
                print(f"KILLED         {name}: {' '.join(hit)}")
            else:
                print(f"KILLED(trap)   {name}: rc={t.returncode} {t.stderr.strip().splitlines()[-1:] }")
    print(f"MUTANTS total={len(MUTANTS)} survived={survived}")
    sys.exit(1 if survived else 0)


if __name__ == "__main__":
    main()
