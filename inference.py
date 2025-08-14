import argparse
from pathlib import Path
import soundfile as sf
import torch
import yaml

from bmfwf.models.bdmfmvdr import BDMFMVDR
from bmfwf.models.bilatdmfmvdr import BilatDMFMVDR
from bmfwf.models.directfiltering import DirectFiltering

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def main(args):
    """Main function for inference."""
    trained_models_dir = Path("bmfwf/trained_models")
    model_dir = trained_models_dir / args.model

    if not model_dir.exists():
        print(f"Model '{args.model}' not found.")
        print("Available models:")
        for model_name in sorted(
            p.name for p in trained_models_dir.iterdir() if p.is_dir()
        ):
            print(f"- {model_name}")
        return

    # Load config
    with open(model_dir / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Determine model class
    if "bilat" in args.model:
        model_class = BilatDMFMVDR
    elif "stwf" in args.model:
        model_class = BDMFMVDR
    elif "df" in args.model:
        model_class = DirectFiltering
    else:
        raise ValueError(f"Could not determine model class for '{args.model}'")

    # Load model from checkpoint
    ckpt_path = next(model_dir.glob("*.ckpt"))
    model = model_class.load_from_checkpoint(
        checkpoint_path=str(ckpt_path),
        **config["model"]["init_args"],
        map_location=device,
    )
    model.eval()

    # Load audio
    noisy, sr = sf.read(args.input, dtype="float32", always_2d=True)
    if sr != model.fs:
        raise ValueError(
            f"Sampling rate of input file ({sr} Hz) does not match model's ({model.fs} Hz)."
        )
    noisy = noisy.T
    # channels need to be ordered as all left, all right (here: left, left, right, right)
    noisy = noisy[(0, 2, 1, 3), :]

    noisy = torch.from_numpy(noisy).unsqueeze(0).to(device)
    batch = {"input": noisy}

    # Run inference
    with torch.no_grad():
        output = model(batch)
    enhanced = output["input_proc"]

    # Save output
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.input).with_name(
            Path(args.input).name.replace(".wav", "") + "_enhanced.wav"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), enhanced.squeeze().T.cpu().numpy(), sr)
    print(f"Enhanced audio saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference script for speech enhancement models."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Name of the pretrained model to use.",
        default="stwf_noCommonSTCM_noRTF",
        choices=[  # see Table II in [10.1109/TASLPRO.2025.3548454]
            "stwf_noCommonSTCM_noRTF",
            "stwf_CommonSTCM_noRTF",
            "stwf_CommonSTCM_globalRTF",
            "stwf_CommonSTCM_ipsiRTF",
            "stwf_bilat_CommonSTCM_noRTF",
            "stwf_bilat_CommonSTCM_global",
            "df_noRTF",
        ],
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to the input audio file (must be binaural (2 channels per side)).",
        default="data/noisy.wav",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the enhanced audio file. If empty, save in the same directory as the input file.",
        default="",
    )

    cli_args = parser.parse_args()
    main(cli_args)
