"use client";

import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { FileIcon } from "lucide-react";

export default function CvDropzone() {
  const [file, setFile] = useState<File | null>(null);
  const [score, setScore] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const edgeURL =
    process.env.NEXT_PUBLIC_EDGE_URL ?? "http://localhost:9000";

  async function handleUpload() {
    if (!file) return;
    setLoading(true);
    setScore(null);
    setError(null);

    try {
      const form = new FormData();
      form.append("file", file);

      const res = await fetch(`${edgeURL}/upload`, {
        method: "POST",
        headers: {
          "X-API-Key": process.env.NEXT_PUBLIC_API_KEY ?? "",
        },
        body: form,
      });

      if (!res.ok) {
        const msg = `Edge error ${res.status}`;
        console.error(msg);
        setError(msg);
        setLoading(false);
        return;
      }

      const json = await res.json(); // {proba, pred}
      setScore(json.proba as number);
    } catch (e: any) {
      console.error(e);
      setError("Falha de rede ou CORS");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div
      className="w-full max-w-lg p-8 border-2 border-dashed rounded-2xl text-center space-y-6"
      onDragOver={(e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = "copy";
      }}
      onDrop={(e) => {
        e.preventDefault();
        const f = e.dataTransfer.files?.[0];
        if (f) setFile(f);
      }}
    >
      <label className="block cursor-pointer">
        <input
          type="file"
          accept=".pdf,.doc,.docx,.txt"
          className="hidden"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        <div className="flex flex-col items-center gap-2">
          <FileIcon className="h-8 w-8" />
          <span>
            {file ? file.name : "Arraste ou clique para escolher um arquivo"}
          </span>
        </div>
      </label>

      <Button disabled={!file || loading} onClick={handleUpload}>
        {loading ? "Enviando…" : "Enviar CV"}
      </Button>

      {loading && <Progress value={75} />} {/* mero placeholder */}

      {score !== null && (
        <p className="text-xl">
          Probabilidade de match:&nbsp;
          <span className="font-bold">{(score * 100).toFixed(1)} %</span>
        </p>
      )}
      {error && (
        <p className="text-red-600">
          {error}
        </p>
      )}
    </div>
  );
}
