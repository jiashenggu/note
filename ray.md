kill node by ip

```bash
ip=100.65.179.138
python3 - <<EOF
import sys, ray, subprocess
ray.init(address="auto")
for n in ray.nodes():
    if n["Alive"] and n["NodeManagerAddress"] == "$ip":
        node_id = n["NodeID"]
        print(f"kicking $ip  node_id={node_id}")
        @ray.remote(num_cpus=0)
        def _die(): subprocess.run(["ray","stop","--force"], check=False)
        ray.get(_die.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node_id, soft=False)).remote())
        sys.exit(0)
print("$ip 未找到或已离线")
EOF
```
