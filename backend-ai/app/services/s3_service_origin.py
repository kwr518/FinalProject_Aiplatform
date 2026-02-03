import boto3
import os
from app.core.config import BUCKET_NAME, AWS_ACCESS_KEY, AWS_SECRET_KEY, S3_CONFIG, AWS_REGION

class S3Service:
    def __init__(self):
        # 인증 정보가 없으면 에러를 명시적으로 출력합니다.
        if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
            print("❌ 에러: AWS 인증 정보를 찾을 수 없습니다. .env 파일을 확인하세요.")

        self.client = boto3.client('s3',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION,  # 👈 리전 명시 추가
            config=S3_CONFIG
        )
        self.bucket = BUCKET_NAME

    def get_presigned_url(self, key):
        return self.client.generate_presigned_url(
            'get_object',
            Params={'Bucket': self.bucket, 'Key': key},
            ExpiresIn=3600
        )

    def download_file(self, key, local_path):
        self.client.download_file(self.bucket, key, local_path)

    def upload_file(self, local_path, key):
        self.client.upload_file(local_path, self.bucket, key)
        
    def delete_file(self, s3_key):
        try:
            print(f"🗑️ Deleting from S3: {s3_key}")
            self.s3.delete_object(Bucket=self.bucket_name, Key=s3_key)
            print(f"✅ S3 Delete Success: {s3_key}")
            return True
        except Exception as e:
            print(f"❌ S3 Delete Failed: {e}")
            return False    

s3_manager = S3Service()