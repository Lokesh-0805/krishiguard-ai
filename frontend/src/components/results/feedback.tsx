"use client";

import { useState } from "react";
import { Check, Loader2, MessageSquareHeart, ThumbsDown, ThumbsUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { ApiError, sendFeedback } from "@/lib/api";
import { useDiseases } from "@/lib/hooks";
import { useLang } from "@/lib/i18n";

export function FeedbackPanel({ predictionId }: { predictionId: string }) {
  const { t, lang } = useLang();
  const [correct, setCorrect] = useState<boolean | null>(null);
  const [helpful, setHelpful] = useState<boolean | null>(null);
  const [suggested, setSuggested] = useState("");
  const [comment, setComment] = useState("");
  const [state, setState] = useState<"idle" | "sending" | "done">("idle");
  const [error, setError] = useState<string | null>(null);
  const diseases = useDiseases(lang);

  const submit = async () => {
    setState("sending"); setError(null);
    try {
      await sendFeedback({
        prediction_id: predictionId, prediction_correct: correct, advisory_helpful: helpful,
        suggested_class: correct === false && suggested ? suggested : null, comment: comment.trim() || null,
      });
      setState("done");
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed"); setState("idle");
    }
  };

  if (state === "done") {
    return (
      <Card className="no-print"><CardContent className="flex items-center gap-3 text-sm" data-testid="feedback-done">
        <Check className="size-5 text-success" />{t("fb_thanks")}
      </CardContent></Card>
    );
  }
  return (
    <Card className="no-print">
      <CardContent className="space-y-4">
        <h2 className="flex items-center gap-2 font-semibold"><MessageSquareHeart className="size-4 text-primary" />{t("feedback_title")}</h2>
        <div className="flex flex-wrap gap-2">
          <Button variant={correct === true ? "default" : "outline"} size="sm" onClick={() => setCorrect(correct === true ? null : true)}><ThumbsUp />{t("fb_correct")}</Button>
          <Button variant={correct === false ? "default" : "outline"} size="sm" onClick={() => setCorrect(correct === false ? null : false)}><ThumbsDown />{t("fb_incorrect")}</Button>
          <Button variant={helpful === true ? "default" : "outline"} size="sm" onClick={() => setHelpful(helpful ? null : true)}><Check />{t("fb_helpful")}</Button>
        </div>
        {correct === false && (
          <select value={suggested} onChange={(e) => setSuggested(e.target.value)} className="h-9 w-full rounded-lg border border-input bg-background px-3 text-sm">
            <option value="">—</option>
            {diseases.data?.map((d) => <option key={d.id} value={d.id}>{d.crop} — {d.name}</option>)}
          </select>
        )}
        <Textarea value={comment} onChange={(e) => setComment(e.target.value.slice(0, 1000))} placeholder={t("fb_comment")} rows={2} />
        {error && <p className="text-sm text-destructive">{error}</p>}
        <Button size="sm" onClick={submit} disabled={state === "sending" || (correct === null && helpful === null && !comment.trim())}>
          {state === "sending" && <Loader2 className="animate-spin" />}{t("fb_send")}
        </Button>
      </CardContent>
    </Card>
  );
}
