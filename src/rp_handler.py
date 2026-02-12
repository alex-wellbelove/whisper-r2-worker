"""
rp_handler.py for runpod worker

Modified to support R2/S3 output for large transcription results.

Environment variables for R2 output:
- R2_ENDPOINT_URL: R2 endpoint (e.g., https://xxx.r2.cloudflarestorage.com)
- R2_ACCESS_KEY_ID: R2 access key
- R2_SECRET_ACCESS_KEY: R2 secret key
- R2_BUCKET_NAME: R2 bucket name
- R2_PUBLIC_URL: Public URL prefix for the bucket (e.g., https://pub-xxx.r2.dev)

If these are set, results will be uploaded to R2 and a URL returned.
Can also pass s3Config in the request input to override.
"""
import base64
import json
import os
import tempfile
import uuid
from datetime import datetime

from rp_schema import INPUT_VALIDATIONS
from runpod.serverless.utils import download_files_from_urls, rp_cleanup, rp_debugger
from runpod.serverless.utils.rp_validator import validate
import runpod
import predict


MODEL = predict.Predictor()
MODEL.setup()

# Check for R2 config in environment
R2_CONFIG = None
if os.environ.get('R2_ENDPOINT_URL') and os.environ.get('R2_BUCKET_NAME'):
    R2_CONFIG = {
        'endpoint_url': os.environ.get('R2_ENDPOINT_URL'),
        'access_key_id': os.environ.get('R2_ACCESS_KEY_ID'),
        'secret_access_key': os.environ.get('R2_SECRET_ACCESS_KEY'),
        'bucket_name': os.environ.get('R2_BUCKET_NAME'),
        'public_url': os.environ.get('R2_PUBLIC_URL', ''),
    }
    print(f"R2 output configured: bucket={R2_CONFIG['bucket_name']}")


def upload_to_r2(data: dict, config: dict, job_id: str) -> str:
    """
    Upload JSON data to R2 and return the public URL.
    
    Parameters:
    data (dict): Data to upload as JSON
    config (dict): R2 configuration
    job_id (str): Job ID for filename
    
    Returns:
    str: Public URL to the uploaded file
    """
    import boto3
    from botocore.config import Config
    
    s3_client = boto3.client(
        's3',
        endpoint_url=config['endpoint_url'],
        aws_access_key_id=config['access_key_id'],
        aws_secret_access_key=config['secret_access_key'],
        config=Config(signature_version='s3v4'),
    )
    
    # Generate filename with timestamp
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    filename = f"transcriptions/{timestamp}_{job_id}.json"
    
    # Upload
    json_data = json.dumps(data, ensure_ascii=False, indent=2)
    s3_client.put_object(
        Bucket=config['bucket_name'],
        Key=filename,
        Body=json_data.encode('utf-8'),
        ContentType='application/json',
    )
    
    # Return public URL
    if config.get('public_url'):
        return f"{config['public_url'].rstrip('/')}/{filename}"
    else:
        return f"{config['endpoint_url']}/{config['bucket_name']}/{filename}"


def base64_to_tempfile(base64_file: str) -> str:
    '''
    Convert base64 file to tempfile.

    Parameters:
    base64_file (str): Base64 file

    Returns:
    str: Path to tempfile
    '''
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(base64.b64decode(base64_file))

    return temp_file.name


@rp_debugger.FunctionTimer
def run_whisper_job(job):
    '''
    Run inference on the model.

    Parameters:
    job (dict): Input job containing the model parameters

    Returns:
    dict: The result of the prediction, or URL to results if R2 configured
    '''
    job_input = job['input']
    job_id = job.get('id', str(uuid.uuid4()))

    with rp_debugger.LineTimer('validation_step'):
        input_validation = validate(job_input, INPUT_VALIDATIONS)

        if 'errors' in input_validation:
            return {"error": input_validation['errors']}
        job_input = input_validation['validated_input']

    if not job_input.get('audio', False) and not job_input.get('audio_base64', False):
        return {'error': 'Must provide either audio or audio_base64'}

    if job_input.get('audio', False) and job_input.get('audio_base64', False):
        return {'error': 'Must provide either audio or audio_base64, not both'}

    if job_input.get('audio', False):
        with rp_debugger.LineTimer('download_step'):
            audio_input = download_files_from_urls(job['id'], [job_input['audio']])[0]

    if job_input.get('audio_base64', False):
        audio_input = base64_to_tempfile(job_input['audio_base64'])

    with rp_debugger.LineTimer('prediction_step'):
        whisper_results = MODEL.predict(
            audio=audio_input,
            model_name=job_input["model"],
            transcription=job_input["transcription"],
            translation=job_input["translation"],
            translate=job_input["translate"],
            language=job_input["language"],
            temperature=job_input["temperature"],
            best_of=job_input["best_of"],
            beam_size=job_input["beam_size"],
            patience=job_input["patience"],
            length_penalty=job_input["length_penalty"],
            suppress_tokens=job_input.get("suppress_tokens", "-1"),
            initial_prompt=job_input["initial_prompt"],
            condition_on_previous_text=job_input["condition_on_previous_text"],
            temperature_increment_on_fallback=job_input["temperature_increment_on_fallback"],
            compression_ratio_threshold=job_input["compression_ratio_threshold"],
            logprob_threshold=job_input["logprob_threshold"],
            no_speech_threshold=job_input["no_speech_threshold"],
            enable_vad=job_input["enable_vad"],
            word_timestamps=job_input["word_timestamps"]
        )

    with rp_debugger.LineTimer('cleanup_step'):
        rp_cleanup.clean(['input_objects'])

    # Strip bulky fields from segments to reduce payload size
    # (tokens, words, etc. aren't needed for SRT/text output)
    if 'segments' in whisper_results:
        keep_keys = {'id', 'start', 'end', 'text', 'seek', 'temperature',
                      'avg_logprob', 'compression_ratio', 'no_speech_prob'}
        whisper_results['segments'] = [
            {k: v for k, v in seg.items() if k in keep_keys}
            for seg in whisper_results['segments']
        ]

    # Check for R2 config - from request input or environment
    r2_config = None
    if job_input.get('s3Config'):
        # Config from request
        s3_cfg = job_input['s3Config']
        r2_config = {
            'endpoint_url': s3_cfg.get('endpointUrl'),
            'access_key_id': s3_cfg.get('accessId'),
            'secret_access_key': s3_cfg.get('accessSecret'),
            'bucket_name': s3_cfg.get('bucketName'),
            'public_url': s3_cfg.get('publicUrl', ''),
        }
    elif R2_CONFIG:
        # Config from environment
        r2_config = R2_CONFIG
    
    # Upload to R2 if configured
    if r2_config and r2_config.get('endpoint_url') and r2_config.get('bucket_name'):
        try:
            with rp_debugger.LineTimer('r2_upload_step'):
                result_url = upload_to_r2(whisper_results, r2_config, job_id)
            
            # Return minimal response with URL
            return {
                'status': 'completed',
                'result_url': result_url,
                'detected_language': whisper_results.get('detected_language'),
                'model': whisper_results.get('model'),
                'segment_count': len(whisper_results.get('segments', [])),
            }
        except Exception as e:
            print(f"R2 upload failed: {e}")
            # Fall back to returning full results
            return whisper_results
    
    return whisper_results


runpod.serverless.start({"handler": run_whisper_job})
