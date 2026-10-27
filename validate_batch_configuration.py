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
    assert len(jobs)==816, len(jobs)
    runner._validate_strict_batch_order(jobs)

    fake=[j for j in jobs if j.profile_name=="fake_quebec_statevector"]
    assert sorted({j.num_qubits for j in fake})==[2,3,4,5]
    assert all(j.backend=="fake_statevector" and j.batch_phase==1 for j in fake)
    first_fake=min(j.index for j in fake)-1
    assert all(j.batch_phase==0 for j in jobs[:first_fake])
    assert all(j.batch_phase==1 for j in jobs[first_fake:])

    normal=[j for j in jobs if j.batch_phase==0]
    assert normal
    assert all(j.backend=="aer_simulator" for j in normal)
    small=[j for j in normal if j.profile_name=="aer_small_constant_depolarizing"]
    assert len(small)==207, len(small)
    assert all(j.noise=="small_constant_depolarizing" for j in small)
    assert all(j.report_tag=="GPU_small_constant_depolarizing_0p2pct" for j in small)
    normal_rounds=[j.depth_round for j in normal]
    assert normal_rounds==sorted(normal_rounds)
    fake_rounds=[j.depth_round for j in fake]
    assert fake_rounds==sorted(fake_rounds)

    # Every normal round is globally complete before the next round begins.
    normal_groups=runner._group_jobs_by_stage(normal)
    assert [key for key,_ in normal_groups]==[(0,r) for r in range(5)]
    expected_normal_profiles={
        "aer_small_constant_depolarizing",
        "aer_increasing_pauli",
        "aer_increasing_pauli_amplitude_damping",
    }
    for (phase, round_index), members in normal_groups:
        assert phase==0
        assert {j.profile_name for j in members}==expected_normal_profiles
        assert all(j.depth_round==round_index for j in members)
    for j in jobs:
        expected_fold=(1,3,5) if j.num_qubits==5 else (1,3,5,7)
        assert j.fold_factors==expected_fold,(j.index,j.fold_factors)
    assert sorted({j.depth for j in jobs if j.unitary=="hea_crz" and j.num_qubits==2})==[30,60,90,120]
    assert u.get_unitary_max_depth("random_type",5)>=128
    r=Recorder(); prepare_initial_state(r,list(range(5)))
    assert len(r.operations)==14
    assert r.operations[-4:]==[("cx",0,1),("cx",1,2),("cx",2,3),("cx",3,4)]
    print("PASS: 816 unique jobs (621 normal + 195 final fake)")
    print("PASS: strict order = complete normal-Aer round 1/2/3/4/5, then complete FakeQuebec rounds")
    print("PASS: FakeQuebec statevector covers every 2-5q point")
    print("PASS: 5q preparation = 5 RY + 5 RZ + 4-CX chain")
    print("Jobs by profile:",dict(Counter(j.profile_name for j in jobs)))
    print("Jobs by qubit:",dict(sorted(Counter(j.num_qubits for j in jobs).items())))
    print("First 12 jobs:")
    for j in jobs[:12]: print(j.index,j.batch_phase,j.depth_round,j.profile_name,j.method,j.num_qubits,j.unitary,j.depth)
    print("First 6 fake jobs:")
    for j in fake[:6]: print(j.index,j.batch_phase,j.depth_round,j.method,j.num_qubits,j.unitary,j.depth)
    if warnings:
        print("Warnings:")
        for w in dict.fromkeys(warnings): print(" -",w)
    return 0
if __name__=="__main__": raise SystemExit(main())
