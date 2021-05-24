extern crate anyhow;

use rust_bert::gpt_neo::{
    GptNeoConfigResources, GptNeoMergesResources, GptNeoModelResources, GptNeoVocabResources,
};
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::text_generation::{TextGenerationConfig, TextGenerationModel};
use rust_bert::resources::{RemoteResource, Resource};
use tch::Device;

fn main() -> anyhow::Result<()> {
    //    Set-up model resources
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        GptNeoConfigResources::GPT_NEO_1_3B,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        GptNeoVocabResources::GPT_NEO_1_3B,
    ));
    let merges_resource = Resource::Remote(RemoteResource::from_pretrained(
        GptNeoMergesResources::GPT_NEO_1_3B,
    ));
    let model_resource = Resource::Remote(RemoteResource::from_pretrained(
        GptNeoModelResources::GPT_NEO_1_3B,
    ));
    let generate_config = TextGenerationConfig {
        model_type: ModelType::GPTNeo,
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource,
        min_length: 10,
        max_length: 32,
        do_sample: false,
        early_stopping: true,
        num_beams: 4,
        num_return_sequences: 1,
        device: Device::Cpu,
        ..Default::default()
    };

    let model = TextGenerationModel::new(generate_config)?;

    let input_context_1 = "It was a very nice and sunny";
    let input_context_2 = "It was a gloom winter night, and";
    let output = model.generate(&[input_context_1, input_context_2], None);

    for sentence in output {
        println!("{}", sentence);
    }
    Ok(())
}