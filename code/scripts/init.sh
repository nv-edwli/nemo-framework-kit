echo "Initializing the Llama-2 ${1} model."

if [ ! -f /project/models/llama2-$1.nemo ]; then
    echo "Cloning the model weights onto underlying host system via /project/models host mount."
    echo ""
    echo "Working..."
    python /project/code/scripts/download.py $1 

    echo "Converting the model weights to NeMo format. May take a few moments to complete."
    echo ""
    echo "Working..."
    python /opt/NeMo/scripts/checkpoint_converters/convert_llama_hf_to_nemo.py --input_name_or_path=/project/models/llama2-$1-hf/ --output_path=/project/models/llama2-$1.nemo
    echo "llama2-${1}.nemo should be ready now. "
else
    echo "llama2-${1}.nemo already exists. Skipping."
fi

sleep 2
echo "Llama-2 ${1} model is initialized."