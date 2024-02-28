from src.llm_runner import DefaultLLMRunner, MedusaRunner

def build_runner(args):
    if args.use_medusa:
        return MedusaRunner(args)
    else:
        return DefaultLLMRunner(args)