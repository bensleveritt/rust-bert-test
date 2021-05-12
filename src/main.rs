extern crate anyhow;

use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::text_generation::{TextGenerationConfig, TextGenerationModel};

fn main() -> anyhow::Result<()> {
    //    Set-up masked LM model
    let generate_config = TextGenerationConfig {
        model_type: ModelType::GPT2,
        max_length: 30,
        do_sample: true,
        num_beams: 5,
        temperature: 1.1,
        num_return_sequences: 1,
        ..Default::default()
    };
    let model = TextGenerationModel::new(generate_config)?;

    let input_context = "The dog can";
    let second_input_context = "The cat was";
    let output = model.generate(&[input_context, second_input_context], None);

    for sentence in output {
        println!("{:?}", sentence);
    }
    Ok(())
}
