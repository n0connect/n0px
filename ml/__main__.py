#!/usr/bin/env python3
"""
ML Main Entry Point
===================
Command-line interface and application initialization.
"""

import sys
import argparse
import logging

from . import config
from . import utils
from . import core
from . import models
from . import data
from . import trainer as trainer_module
from . import tests

logger = utils.setup_logger(__name__, config.LOG_LEVEL)


# ============================================================
#  CLI COMMANDS
# ============================================================
def cmd_bootstrap(args) -> int:
    """
    Run bootstrap test suite.
    Tests gRPC connection, data integrity, mathematical validation, etc.
    """
    utils.print_banner("BOOTSTRAP TEST MODE")
    
    # Connect to Go Bridge
    print()
    logger.info("Connecting to Go Bridge at " + config.GRPC_SERVER_ADDRESS)
    bridge = core.create_and_connect_bridge()
    
    if not bridge:
        utils.print_err("Failed to connect to Go Bridge")
        return 1
    
    try:
        # Run test suite
        test_suite = tests.BootstrapTestSuite(bridge)
        results = test_suite.run_all()
        
        # Return exit code based on results
        return 0 if results["all_passed"] else 1
    
    finally:
        bridge.disconnect()


def cmd_stream_test(args) -> int:
    """
    Test continuous streaming from Go Bridge.
    """
    utils.print_banner("STREAM TEST MODE")
    
    bridge = core.create_and_connect_bridge()
    if not bridge:
        utils.print_err("Failed to connect to Go Bridge")
        return 1
    
    try:
        utils.print_section("Streaming Data Test")
        logger.info(f"Receiving {args.count} packets...")
        
        packets = bridge.get_packets(args.count)
        
        logger.info(f"Received {len(packets)} packets")
        
        # Show statistics
        labels_dist = {}
        for pkt in packets:
            if pkt.label not in labels_dist:
                labels_dist[pkt.label] = 0
            labels_dist[pkt.label] += 1
        
        utils.print_section("Label Distribution")
        for label_id in sorted(labels_dist.keys()):
            label_name = utils.format_label(label_id)
            count = labels_dist[label_id]
            pct = count / len(packets) * 100
            print(f"  {label_name}: {count} ({pct:.1f}%)")
        
        return 0
    
    finally:
        bridge.disconnect()


def cmd_train(args) -> int:
    """
    Train the ML model on streaming data from Go Bridge.
    Uses dynamic model discovery - specify model by name via --model flag.
    """
    utils.print_banner("TRAINING MODE")
    
    # List available models if requested
    if args.list_models:
        available = models.ModelRegistry.list()
        if not available:
            print("No models registered. Add models to ml/models/")
            return 0
        print("\nAvailable models:")
        for name in sorted(available.keys()):
            print(f"  - {name}")
        return 0
    
    bridge = core.create_and_connect_bridge()
    if not bridge:
        utils.print_err("Failed to connect to Go Bridge")
        return 1
    
    try:
        # Initialize components
        logger.info("Initializing training components...")
        
        # Create model from registry
        try:
            model = models.ModelRegistry.get(
                args.model,
                input_dim=config.MODEL_INPUT_DIM,
                hidden_dim=config.MODEL_HIDDEN_DIM,
                num_layers=config.MODEL_NUM_LAYERS,
                dropout=config.MODEL_DROPOUT,
            )
        except ValueError as e:
            utils.print_err(str(e))
            logger.error(f"Model '{args.model}' not found")
            return 1
        
        # Create data iterator
        data_iter = data.StreamDataIterator(
            bridge=bridge,
            batch_size=args.batch_size,
            balance_labels=True,
        )
        
        # Create trainer
        tr = trainer_module.Trainer(
            model=model,
            data_iterator=data_iter,
            learning_rate=config.MODEL_LEARNING_RATE,
            checkpoint_dir=args.checkpoint_dir,
        )
        
        # Train
        results = tr.train(
            num_epochs=args.epochs,
            batches_per_epoch=args.batches_per_epoch,
            val_every=args.val_every,
            save_every=args.save_every,
        )
        
        utils.print_ok(f"Training completed in {results['total_time']:.2f}s")
        logger.info(f"Best validation accuracy: {results['best_accuracy']:.4f}")
        
        return 0
    
    except Exception as e:
        utils.print_err(f"Training failed: {e}")
        logger.exception("Training error")
        return 1
    
    finally:
        bridge.disconnect()


def cmd_inference(args) -> int:
    """
    Run inference on streaming data.
    Requires --model-type to specify which model class to instantiate.
    """
    utils.print_banner("INFERENCE MODE")
    
    # Instantiate model from registry
    logger.info(f"Loading model type '{args.model_type}' from {args.model_path}...")
    try:
        model = models.ModelRegistry.get(args.model_type)
        model.load(args.model_path)
    except (FileNotFoundError, ValueError) as e:
        utils.print_err(f"Failed to load model: {e}")
        return 1
    
    # Connect to bridge
    bridge = core.create_and_connect_bridge()
    if not bridge:
        utils.print_err("Failed to connect to Go Bridge")
        return 1
    
    try:
        inference = trainer_module.InferenceEngine(model)
        
        utils.print_section("Running Inference")
        logger.info(f"Processing {args.count} packets...")
        
        correct = 0
        total = 0
        
        for i in range(args.count):
            pkt = bridge.get_packet()
            if pkt is None:
                break
            
            result = inference.predict(pkt)
            
            if result["is_correct"]:
                correct += 1
            total += 1
            
            if (i + 1) % 50 == 0 or i == args.count - 1:
                accuracy = correct / total * 100
                logger.info(
                    f"Progress: {i + 1}/{args.count}, "
                    f"Accuracy: {accuracy:.2f}%"
                )
        
        if total > 0:
            accuracy = correct / total * 100
            utils.print_ok(f"Inference complete - Accuracy: {accuracy:.2f}%")
        
        return 0
    
    except Exception as e:
        utils.print_err(f"Inference failed: {e}")
        return 1
    
    finally:
        bridge.disconnect()


def cmd_analyze(args) -> int:
    """
    Analyze data from Go Bridge without training.
    """
    utils.print_banner("DATA ANALYSIS MODE")
    
    bridge = core.create_and_connect_bridge()
    if not bridge:
        utils.print_err("Failed to connect to Go Bridge")
        return 1
    
    try:
        utils.print_section("Fetching Data")
        logger.info(f"Collecting {args.samples} packets for analysis...")
        
        packets = bridge.get_packets(args.samples)
        logger.info(f"Collected {len(packets)} packets")
        
        utils.print_section("Statistical Analysis")
        stats = tests.StatisticalTests.analyze_packets(packets)
        
        logger.info(f"Monobit z-score: {stats['monobit']['z_score']:.3f}")
        logger.info(f"Gaussian mean: {stats['gaussian']['mean']:.4f}")
        logger.info(f"Gaussian std: {stats['gaussian']['std']:.4f}")
        logger.info(f"Autocorrelation: {stats['autocorrelation']['correlation']:.4f}")
        
        return 0
    
    finally:
        bridge.disconnect()


def cmd_serve(args) -> int:
    """
    Run ML service in continuous streaming mode.
    Connects to Go Bridge, streams data, and performs real-time inference.
    Useful for production deployment or continuous monitoring.
    """
    utils.print_banner("ML SERVICE - CONTINUOUS STREAMING MODE")
    
    # Load model if provided
    model = None
    if args.model_path:
        utils.print_section("Loading Model")
        try:
            logger.info(f"Loading model from {args.model_path}...")
            model = models.ModelRegistry.get(args.model_type or "nn_classifier")
            model.load(args.model_path)
            logger.info("Model loaded successfully")
        except FileNotFoundError as e:
            utils.print_err(f"Model not found: {e}")
            logger.warning("Continuing without model (inference disabled)")
        except Exception as e:
            utils.print_err(f"Failed to load model: {e}")
            logger.warning("Continuing without model (inference disabled)")
    
    # Connect to bridge
    utils.print_section("Connecting to Go Bridge")
    bridge = core.create_and_connect_bridge()
    if not bridge:
        utils.print_err("Failed to connect to Go Bridge")
        return 1
    
    logger.info(f"Connected to {config.GRPC_SERVER_ADDRESS}")
    
    try:
        utils.print_section("Starting Streaming Service")
        logger.info("Listening for packets (Ctrl+C to stop)...")
        
        if model:
            inference = trainer_module.InferenceEngine(model)
            logger.info("Inference engine ready")
        
        # Statistics
        packet_count = 0
        correct_predictions = 0
        label_dist = {}
        
        import signal
        def signal_handler(sig, frame):
            nonlocal packet_count
            logger.info("Service shutting down...")
            raise KeyboardInterrupt()
        
        signal.signal(signal.SIGINT, signal_handler)
        
        print()
        while True:
            try:
                pkt = bridge.get_packet()
                if pkt is None:
                    continue
                
                packet_count += 1
                
                # Track label distribution
                if pkt.label not in label_dist:
                    label_dist[pkt.label] = 0
                label_dist[pkt.label] += 1
                
                # Run inference if model loaded
                if model:
                    result = inference.predict(pkt)
                    if result["is_correct"]:
                        correct_predictions += 1
                    
                    accuracy = correct_predictions / packet_count * 100 if packet_count > 0 else 0.0
                    pred_label = utils.format_label(result.get("predicted_label", -1))
                    true_label = utils.format_label(pkt.label)
                    
                    status = "✓" if result["is_correct"] else "✗"
                    logger.debug(f"{status} Predicted: {pred_label}, True: {true_label}, Accuracy: {accuracy:.1f}%")
                
                # Print periodic status
                if packet_count % 100 == 0:
                    accuracy_str = ""
                    if model:
                        accuracy = correct_predictions / packet_count * 100
                        accuracy_str = f" | Accuracy: {accuracy:.1f}%"
                    
                    dist_str = " | Distribution: " + ", ".join(
                        f"{utils.format_label(k)}:{v}" for k, v in sorted(label_dist.items())
                    )
                    
                    logger.info(f"Processed {packet_count} packets{dist_str}{accuracy_str}")
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.exception(f"Error processing packet: {e}")
                continue
        
        # Final statistics
        print()
        utils.print_section("Service Statistics")
        logger.info(f"Total packets processed: {packet_count}")
        logger.info("Label distribution:")
        for label_id in sorted(label_dist.keys()):
            label_name = utils.format_label(label_id)
            count = label_dist[label_id]
            pct = count / packet_count * 100 if packet_count > 0 else 0
            logger.info(f"  {label_name}: {count} ({pct:.1f}%)")
        
        if model:
            accuracy = correct_predictions / packet_count * 100 if packet_count > 0 else 0.0
            logger.info(f"Overall accuracy: {accuracy:.2f}%")
        
        utils.print_ok("Service stopped gracefully")
        return 0
    
    except Exception as e:
        utils.print_err(f"Service error: {e}")
        logger.exception("Unexpected error")
        return 1
    
    finally:
        bridge.disconnect()


# ============================================================
#  MAIN CLI
# ============================================================
def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PRIME-X ML Pipeline - Real-time prime classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ml bootstrap                # Run bootstrap tests
  python -m ml stream --count 1000     # Stream 1000 packets
  python -m ml train --epochs 10        # Train model
  python -m ml inference --model-path ./model.pkl --count 100
  python -m ml analyze --samples 1000   # Analyze data
        """,
    )
    
    # Common arguments
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--grpc-host",
        default=config.GRPC_SERVER_HOST,
        help=f"gRPC server host (default: {config.GRPC_SERVER_HOST})"
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=config.GRPC_SERVER_PORT,
        help=f"gRPC server port (default: {config.GRPC_SERVER_PORT})"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # bootstrap
    subparsers.add_parser(
        "bootstrap",
        help="Run bootstrap test suite"
    )
    
    # stream
    stream_parser = subparsers.add_parser(
        "stream",
        help="Test continuous streaming"
    )
    stream_parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of packets to stream (default: 100)"
    )
    
    # train
    train_parser = subparsers.add_parser(
        "train",
        help="Train ML model"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
        help=f"Number of epochs (default: {config.EPOCHS})"
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help=f"Batch size (default: {config.BATCH_SIZE})"
    )
    train_parser.add_argument(
        "--model",
        default="nn_classifier",
        help="Model to train: nn_classifier, torch_classifier, or custom (default: nn_classifier)"
    )
    train_parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    train_parser.add_argument(
        "--batches-per-epoch",
        type=int,
        default=100,
        help="Batches per epoch (default: 100)"
    )
    train_parser.add_argument(
        "--val-every",
        type=int,
        default=5,
        help="Validate every N epochs (default: 5)"
    )
    train_parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N epochs (default: 10)"
    )
    train_parser.add_argument(
        "--checkpoint-dir",
        default="./checkpoints",
        help="Directory to save checkpoints (default: ./checkpoints)"
    )
    
    # inference
    inference_parser = subparsers.add_parser(
        "inference",
        help="Run inference on streaming data"
    )
    inference_parser.add_argument(
        "--model-path",
        required=True,
        help="Path to trained model"
    )
    inference_parser.add_argument(
        "--model-type",
        default="nn_classifier",
        help="Model type to instantiate (default: nn_classifier)"
    )
    inference_parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of packets to inference (default: 100)"
    )
    
    # analyze
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze data from Go Bridge"
    )
    analyze_parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of samples to analyze (default: 1000)"
    )
    
    # serve
    serve_parser = subparsers.add_parser(
        "serve",
        help="Run ML service in continuous streaming mode"
    )
    serve_parser.add_argument(
        "--model-path",
        default=None,
        help="Optional path to pre-trained model for continuous inference"
    )
    serve_parser.add_argument(
        "--model-type",
        default="nn_classifier",
        help="Model type to instantiate (default: nn_classifier)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        log_level = "DEBUG"
    else:
        log_level = config.LOG_LEVEL
    
    logger = utils.setup_logger(__name__, log_level)
    
    # Update config with CLI args
    config.GRPC_SERVER_HOST = args.grpc_host
    config.GRPC_SERVER_PORT = args.grpc_port
    
    # Run command
    if args.command == "bootstrap":
        return cmd_bootstrap(args)
    elif args.command == "stream":
        return cmd_stream_test(args)
    elif args.command == "train":
        return cmd_train(args)
    elif args.command == "inference":
        return cmd_inference(args)
    elif args.command == "analyze":
        return cmd_analyze(args)
    elif args.command == "serve":
        return cmd_serve(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
