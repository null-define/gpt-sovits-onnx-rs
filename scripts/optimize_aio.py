import os
import glob
import argparse
from pathlib import Path
import logging
import onnx
from onnx import version_converter
from onnxsim import simplify
import onnxruntime
from onnxoptimizer import (
    optimize,
    get_fuse_and_elimination_passes,
)
from onnxruntime.transformers import float16
from onnxruntime.quantization import quantize_dynamic, QuantType

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize ONNX models.")
    parser.add_argument(
        "--input-dir",
        default="onnx/kaoyu",
        help="Input directory containing ONNX files",
    )
    parser.add_argument(
        "--output-dir",
        default="onnx-patched/kaoyu",
        help="Output directory for optimized models",
    )
    return parser.parse_args()


def validate_environment():
    try:
        import onnxruntime
    except ImportError:
        logger.error(
            "onnxruntime is not installed. Install it with 'pip install onnxruntime'"
        )
        exit(1)
    try:
        import onnxslim
    except ImportError:
        logger.error(
            "onnxslim is not installed. Install it with 'pip install onnxslim'"
        )
        exit(1)


def process_model(file_path, output_path):
    logger.info(f"Processing model: {file_path}")
    model = onnx.load(file_path)
    
    # convert to fp6 with these 2 models because they are huge
    # if "g2p" in file_path.lower():
    #     model = float16.convert_float_to_float16(model, keep_io_types=True)
    #     logger.info(f"FP16 conversion done for: {output_path}")
    #     if "g2p" in file_path.lower():
    #         onnx.save(model, output_path)  # Ensure model is saved
    #         return output_path
    model = version_converter.convert_version(model, 21)
    logger.info(f"Opset conversion done for: {output_path}")
    # change and simplify except vits because vits model has bug
    if "vits" not in file_path.lower():
        model, _ = simplify(model)
        logger.info(f"ONNX simplification done for: {output_path}")
    # optimize for all
    model = optimize(
        model=model,
        passes=get_fuse_and_elimination_passes(),
    )
    # slim for all
    from onnxslim import slim, OptimizationSettings
    model = slim(model)
    logger.info(f"ONNX slim optimization done for: {output_path}")

    # quant decoder to int8 && save
    if "decoder" in output_path.lower() or "g2p" in output_path.lower() :
        model = quantize_dynamic(model, output_path)
        logger.info(f"INT8 quant done for: {output_path}")
    else:
        onnx.save(model, output_path)  # Ensure model is saved


    # use ort on-device optimizer to get better performance
    # import onnxruntime as rt
    # sess_options = rt.SessionOptions()
    # # Set graph optimization level
    # sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    # # To enable model serialization after graph optimization set this
    # sess_options.optimized_model_filepath = output_path
    # session = rt.InferenceSession(output_path, sess_options)

    return output_path


def main():
    args = parse_args()
    validate_environment()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Find ONNX files
    onnx_files = glob.glob(os.path.join(args.input_dir, "*.onnx"))
    if not onnx_files:
        logger.error(f"No ONNX files found in {args.input_dir}")
        exit(1)

    # Validate required models
    if not any("decoder" in f.lower() for f in onnx_files):
        logger.error("No decoder model found in the folder")
        exit(1)
    if not any("vits" in f.lower() for f in onnx_files):
        logger.error("No vits model found in the folder")
        exit(1)

    # Process each model
    for file_path in onnx_files:
        filename = os.path.basename(file_path)
        output_path = os.path.join(args.output_dir, filename)
        final_path = process_model(file_path, output_path)
        logger.info(f"Optimization complete for: {final_path}")

    logger.info("All models processed successfully!")


if __name__ == "__main__":
    main()
