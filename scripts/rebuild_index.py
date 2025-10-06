import os, argparse, json
import numpy as np
from google.cloud import storage
from src.faiss_ivfpq_utils import train_ivfpq, create_sharded_indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-path', type=str, required=True)
    parser.add_argument('--vectors-path', type=str, required=True)
    parser.add_argument('--ids-path', type=str, required=True)
    parser.add_argument('--out-dir', type=str, default='/tmp/index_out')
    parser.add_argument('--gcs-bucket', type=str, required=True)
    parser.add_argument('--shard-size', type=int, default=50000)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sample = np.load(args.sample_path)
    trained_path = train_ivfpq(sample, index_path=os.path.join(args.out_dir,'trained_ivfpq.index'))
    vectors = np.load(args.vectors_path)
    ids = np.load(args.ids_path)
    shards_dir = os.path.join(args.out_dir,'shards')
    create_sharded_indices(vectors, ids, shard_size=args.shard_size, trained_index_path=trained_path, out_dir=shards_dir)
    client = storage.Client()
    bucket = client.bucket(args.gcs_bucket)
    manifest = {'version':'v1','created_at':None,'shards':[],'meta':None}
    for fname in sorted(os.listdir(shards_dir)):
        local = os.path.join(shards_dir, fname)
        blob_name = f'indexes/{fname}'
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local)
        manifest['shards'].append(f'gs://{args.gcs_bucket}/{blob_name}')
    manifest['created_at'] = __import__('datetime').datetime.utcnow().isoformat()+'Z'
    manifest_blob = bucket.blob('indexes/index_manifest.json')
    manifest_blob.upload_from_string(json.dumps(manifest,indent=2), content_type='application/json')
    print('Uploaded shards and manifest to GCS: gs://{}/indexes/index_manifest.json'.format(args.gcs_bucket))

if __name__ == '__main__':
    main()
