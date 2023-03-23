@testset "encoder -> decode -> original string" begin
    inputs = [
        "hello",
        "big data little comprehension"]

    tokenizer = Diffusers.CLIPTokenizer()
    tokens, pad_mask = Diffusers.tokenize(
        tokenizer, inputs; context_length=7,
        add_start_end=true)

    outputs = [
        strip(Diffusers.decode(tokenizer, @view(tokens[:, i])))
        for i in 1:size(tokens, 2)]
    @test all(inputs .== outputs)
end
