# Robot models the paper ran with

mc_rtc builds both robots from URDFs of the `isri-aist` description packages, in
`~/devel/src/catkin_data_ws/src/` and installed in `~/devel/src/catkin_data_ws/install/share/`.
Both carry a LOCAL change that is committed nowhere upstream: the patches here are those changes,
against the upstream commit named below. The paper's results (2026-09-18) were produced with them;
`paper_ko.lock.json` holds the sha256 of the installed and source URDFs, checked by
`verify_paper_ko.py`.

| package | upstream commit | file | change |
|---|---|---|---|
| `hrp5_p_description` | `40af0c4` | `urdf/HRP5Pmain.urdf` | root-body mass 9.8635 kg -> 1e-6 kg (inertia scaled): the recorded standing load excludes that entry |
| `rhps1_description` | `3931f45e9` | `urdf/RHPS1main_sake2_sake2.urdf` | chest mass 24.326 kg -> 18.7902 kg (inertia scaled): the standing load measured by the foot force sensors is 5.5358 kg below |

To restore them on a fresh checkout:

    cd ~/devel/src/catkin_data_ws/src/hrp5_p_description && git checkout 40af0c4 && git apply <this dir>/hrp5_p_description.patch
    cd ~/devel/src/catkin_data_ws/src/rhps1_description  && git checkout 3931f45e9 && git apply <this dir>/rhps1_description.patch

then rebuild/install the two packages. Edit BOTH the source and the installed copy if a mass is
changed again: mc_rtc reads the installed one. A model change needs the ROUTINE (re-tick), since
the replay bakes the kinematics and dynamics into its cache.
