import os
import glob
import argparse
from pathlib import Path
import logging
import onnx
from onnx import version_converter
from onnxsim import simplify
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxoptimizer import optimize, get_fuse_and_elimination_passes
from onnxconverter_common import float16
from onnxslim import slim
from onnxruntime.transformers import optimizer



# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize ONNX models.")
    parser.add_argument(
        "--input-dir",
        default="onnx/custom",
        help="Input directory containing ONNX files",
    )
    parser.add_argument(
        "--output-dir",
        default="onnx-patched/custom",
        help="Output directory for optimized models",
    )
    parser.add_argument(
        "--no-quant",
        action="store_true",
        help="Disable INT8 quantization for applicable models",
    )
    return parser.parse_args()


def validate_environment():
    """Validate required dependencies."""
    for module, install_name in [
        ("onnxruntime", "onnxruntime"),
        ("onnxslim", "onnxslim"),
    ]:
        try:
            __import__(module)
        except ImportError:
            logger.error(
                f"{install_name} is not installed. Install it with 'pip install {install_name}'"
            )
            exit(1)


def process_model(file_path: str, output_path: str, use_int8_quant: bool) -> str:
    """Process and optimize an ONNX model."""
    logger.info(f"Processing model: {file_path}")
    model = onnx.load(file_path)
    output_lower = output_path.lower()

    # vits model may change and have issue in simplify
    if "vits" in output_lower:
        model = optimize(model, passes=get_fuse_and_elimination_passes())
        logger.info(f"ONNX optimization done for: {output_path}")
        model, _ = simplify(model)
        logger.info(f"ONNX simplification done for: {output_path}")
        # model = slim(model)
        # logger.info(f"ONNX slim optimization done for: {output_path}")
        model = version_converter.convert_version(model, 21)
        onnx.save(model, output_path)
        return output_path

    # Simplify non-BERT models
    if "bert" in output_lower:
        # BERT model optimization

        optimized_model = optimizer.optimize_model(
            model,
            model_type="bert",
            num_heads=16,
            hidden_size=1024,
            only_onnxruntime=True,
        )
        model = version_converter.convert_version(optimized_model.model, 21)
        if use_int8_quant:
            quantize_dynamic(model, output_path)
            logger.info(f"INT8 quantization done for: {output_path}")
        else:
            onnx.save(model, output_path)
        return output_path
    
    if "decoder" in output_lower:
        optimized_model = optimizer.optimize_model(
            model,
            num_heads=24,
            hidden_size=768,
            opt_level=2,
            # only_onnxruntime=True,
        )
        model = optimized_model.model
    
    model = optimize(model, passes=get_fuse_and_elimination_passes())
    logger.info(f"ONNX optimization done for: {output_path}")

    model = slim(model)
    logger.info(f"ONNX slim optimization done for: {output_path}")
    model, _ = simplify(model, include_subgraph=True)
    logger.info(f"ONNX simplification done for: {output_path}")
    
    model = version_converter.convert_version(model, 21)
    logger.info(f"Opset conversion done for: {output_path}")
    # INT8 quantization for specific models, do not quant for vits or ssl
    if use_int8_quant and ("g2p" in file_path.lower() or "decoder" in output_lower):
        quantize_dynamic(
            model,
            output_path,
            op_types_to_quantize=["MatMul", "Attention", "Gather"],
        )
        logger.info(f"INT8 quantization done for: {output_path}")
    else:
        onnx.save(model, output_path)

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
    has_decoder = any("decoder" in f.lower() for f in onnx_files)
    has_vits = any("vits" in f.lower() for f in onnx_files)
    if not has_decoder or not has_vits:
        logger.error(
            f"{'No decoder model found' if not has_decoder else ''}{' and ' if not has_decoder and not has_vits else ''}{'No vits model found' if not has_vits else ''} in the folder"
        )
        exit(1)

    # Process each model
    for file_path in onnx_files:
        output_path = os.path.join(args.output_dir, os.path.basename(file_path))
        final_path = process_model(file_path, output_path, not args.no_quant)
        logger.info(f"Optimization complete for: {final_path}")

    logger.info("All models processed successfully!")


if __name__ == "__main__":
    main()
