from depth_anything_3.api import DepthAnything3

# Initialize and run inference
# model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE").to("cuda")
# # prediction = model.inference(["/data/9katirci/master-automating-pose-estimation/0.jpg"])

# prediction = model.inference(
#     image=["/data/9katirci/master-automating-pose-estimation/0.jpg"],
#     export_dir="./output",
#     export_format="glb"
# )