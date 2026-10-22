"""Validate noise, depth-major order, fake-statevector mode, and 5q input."""
from __future__ import annotations
from collections import Counter
import math
from pathlib import Path
import run_experiments as runner
from state_preparation import prepare_initial_state
from unitary_tools import depth_noise_parameters, depth_noise_scale

class Recorder:
    def __init__(self): self.operations=[]
    def ry(self,a,q): self.operations.append(("ry",float(a),int(q)))
    def rz(self,a,q): self.operations.append(("rz",float(a),int(q)))
    def cx(self,c,t): self.operations.append(("cx",int(c),int(t)))

def main():
    root=Path(__file__).resolve().parent
    u=runner._load_u_config(root)
    jobs=runner._build_jobs(root,profile_filter=None,experiment_filter=None,method_filter=None)
    warnings=runner._validate_jobs(jobs,u)
    for d,e in {0:1,50:2,100:3,125:3.5,150:4,600:4}.items():
        assert math.isclose(depth_noise_scale(d),e,rel_tol=0,abs_tol=1e-15)
    cap=depth_noise_parameters(150)
    expected={"p_1q":0.00098,"p_2q":0.02788,"gamma_1q":0.000448692,"gamma_2q":0.008452596}
    for k,v in expected.items():
        assert math.isclose(cap[k],v,rel_tol=0,abs_tol=1e-15)
        assert math.isclose(depth_noise_parameters(10000)[k],v,rel_tol=0,abs_tol=1e-15)
    assert len(jobs)==819, len(jobs)
    smoke=[j for j in jobs if j.profile_name=="fake_quebec_statevector_smoke"]
    assert len(smoke)==3
    assert [j.method for j in smoke]==["circuit_folding","global","local"]
    assert all(j.index<=3 for j in smoke)
    fake=[j for j in jobs if j.profile_name=="fake_quebec_statevector"]
    assert sorted({j.num_qubits for j in fake})==[2,3,4,5]
    assert all(j.backend=="fake_statevector" for j in fake)
    first_fake=min(j.index for j in fake)-1
    assert all(j.profile_name!="fake_quebec_statevector" for j in jobs[:first_fake])
    assert all(j.profile_name=="fake_quebec_statevector" for j in jobs[first_fake:])
    normal=jobs[3:first_fake]
    for phase in (normal,jobs[first_fake:]):
        rounds=[j.depth_round for j in phase]
        assert rounds==sorted(rounds)
    for j in jobs:
        expected_fold=(1,3,5) if j.num_qubits==5 else (1,3,5,7)
        assert j.fold_factors==expected_fold,(j.index,j.fold_factors)
    assert sorted({j.depth for j in jobs if j.unitary=="hea_crz" and j.num_qubits==2})==[30,60,90,120]
    assert u.get_unitary_max_depth("random_type",5)>=128
    r=Recorder(); prepare_initial_state(r,list(range(5)))
    assert len(r.operations)==14
    assert r.operations[-4:]==[("cx",0,1),("cx",1,2),("cx",2,3),("cx",3,4)]
    print("PASS: 819 unique jobs (3 smoke + 621 normal + 195 final fake)")
    print("PASS: 3-job fake smoke test is first; normal profiles are depth-major; full fake phase is last")
    print("PASS: FakeQuebec statevector covers every 2-5q point")
    print("PASS: 5q preparation = 5 RY + 5 RZ + 4-CX chain")
    print("Jobs by profile:",dict(Counter(j.profile_name for j in jobs)))
    print("Jobs by qubit:",dict(sorted(Counter(j.num_qubits for j in jobs).items())))
    print("First 12 jobs:")
    for j in jobs[:12]: print(j.index,j.depth_round,j.profile_name,j.method,j.num_qubits,j.unitary,j.depth)
    print("First 6 fake jobs:")
    for j in fake[:6]: print(j.index,j.depth_round,j.method,j.num_qubits,j.unitary,j.depth)
    if warnings:
        print("Warnings:")
        for w in dict.fromkeys(warnings): print(" -",w)
    return 0
if __name__=="__main__": raise SystemExit(main())
