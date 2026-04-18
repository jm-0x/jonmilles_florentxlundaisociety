import type { TokenPrediction } from "../types";
import { PredBars } from "./PredBars";

type Props = {
  lensPreds: TokenPrediction[];
  finalPreds: TokenPrediction[] | null;
};

export function PredictionsTab({ lensPreds, finalPreds }: Props) {
  return (
    <div className="flex flex-col gap-5 p-4 overflow-y-auto">
      <section>
        <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-2 select-none">
          logit-lens top-5 at this (layer, token)
        </div>
        <PredBars preds={lensPreds} />
      </section>
      {finalPreds && (
        <section>
          <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-2 select-none">
            actual final prediction (model output at this token)
          </div>
          <PredBars preds={finalPreds} />
        </section>
      )}
    </div>
  );
}
