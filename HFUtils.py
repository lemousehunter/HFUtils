from HFUtilDownload import HFUtilDownload
from HFUtilUpload import HFUtilUpload


class HFUtils(HFUtilDownload, HFUtilUpload):
    pass


if __name__ == "__main__":
    hfutils = HFUtils()
    folder_name = "onnx"
    hfutils.download_folder("BAAI/bge-m3", folder_name, "/Users/mouse/Documents/GitHub/HFUtils/Downloads")
    hfutils.download_config("BAAI/bge-m3", f"/Users/mouse/Documents/GitHub/HFUtils/Downloads/{folder_name}")
    hfutils.download_tokenizer("BAAI/bge-m3", f"/Users/mouse/Documents/GitHub/HFUtils/Downloads/{folder_name}")
    hfutils.upload("lemousehunter/bge-m3-onnx", f"/Users/mouse/Documents/GitHub/HFUtils/Downloads/{folder_name}",
                   replace_if_exist=False)
