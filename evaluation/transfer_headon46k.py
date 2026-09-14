"""Additional 3-9 GPU opponents for the frozen transfer campaign."""
import transfer_campaign as campaign


def main():
    original_spec = campaign.make_spec

    def spec():
        result = original_spec()
        candidates = list(dict.fromkeys(a for a, _ in result['gpu_pairs']))
        opponent = 'headon_iter_46000'
        result['models'] = [m for m in result['models'] if m['name'] in candidates]
        result['models'].append(dict(
            name=opponent, kind='checkpoint', decision_hz=10,
            path=str(campaign.ROOT.parent / 'headon_stage/runs/headon_20k_resume/main/iter_46000.pt')))
        result['gpu_pairs'] = [[a, opponent] for a in candidates]
        result['cpu_pairs'] = []
        return result

    campaign.make_spec = spec
    campaign.OUT = campaign.OUT / 'headon46k_extension'
    campaign.gpu()


if __name__ == '__main__':
    main()
