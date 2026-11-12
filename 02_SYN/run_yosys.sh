#!/bin/sh
# Area and logic-depth estimate with Yosys (no standard-cell library: generic gates only).
#   usage: ./run_yosys.sh          (needs `yosys` on PATH; `yowasp-yosys` also works: YOSYS=yowasp-yosys ./run_yosys.sh)
# Results: build/area_stat.txt (cells per module, "design hierarchy" section) and build/depth_<module>.txt
cd "$(dirname "$0")"
Y=${YOSYS:-yosys}
mkdir -p build
echo "== area (about 1 minute; the depth runs take several more)"
$Y -q area.ys || exit 1
grep -A20 "design hierarchy" build/area_stat.txt | head -22
for top in pe requant_lane engine_ctrl engine_dp fc_unit; do
    sed "s/TOP/$top/g" depth.ys.in > build/depth_$top.ys
    $Y -q build/depth_$top.ys > /dev/null 2>&1
    echo "== depth $top: $(grep -m1 'Longest' build/depth_$top.txt | sed 's/.*length=//; s/).*//') gate levels"
done
