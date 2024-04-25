import os

from .HFUtilBase import HFUtilBase


class HFUtilUpload(HFUtilBase):
    def upload_folder(self, repo_id: str, local_dir: str, replace_if_exist: bool):
        print(self.hf_api)
        print(f"Uploading model {repo_id} to HF...")
        if (self.check_model_exist(repo_id=repo_id) and replace_if_exist) or not self.check_model_exist(repo_id=repo_id):
            # Create repository
            self.hf_api.create_repo(
                repo_id=repo_id,
                token=self._write_token,
                private=False,
                repo_type="model",
                exist_ok=True
            )

            # Upload all files from local_dir
            for root, dirs, files in os.walk(local_dir, topdown=False):
                for name in files:
                    filepath = os.path.join(root, name)
                    filename = "/".join(filepath.split("/")[-2:])
                    print("uploading file: ", filename)

                    self.hf_api.upload_file(
                        path_or_fileobj=filepath,
                        path_in_repo=filename,
                        repo_id=repo_id,
                        repo_type="model",
                        token=self._write_token
                    )
        else:
            print(f"Model {repo_id} already exists. Exiting...")


if __name__ == "__main__":
    hf_util = HFUtilUpload()
    hf_util.upload("lemousehunter/bge-m3-onnx", "/Users/mouse/Documents/GitHub/HFUtils/Downloads/onnx", replace_if_exist=True)