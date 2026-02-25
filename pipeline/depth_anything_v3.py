from depth_anything_3.api import DepthAnything3


# Use the updated, recommended nested model ("-1.1" fix) and
# the same basic inference pattern as in the official README.
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1").to("cuda")

prediction = model.inference(
    image=["/data/9katirci/master-automating-pose-estimation/DJI_20230823160823_0063_D.jpg"],
    # Defaults (process_res=504, process_res_method="upper_bound_resize")
    # already match the examples used in the official repo.
    # You can enable the more accurate but slightly slower ray head like this:
    # use_ray_pose=True,
    export_dir="./output/big",
    export_format="glb",
)