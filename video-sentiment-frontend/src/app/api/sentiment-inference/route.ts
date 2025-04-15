import {
  InvokeEndpointCommand,
  SageMakerRuntimeClient,
} from "@aws-sdk/client-sagemaker-runtime";
import { NextResponse } from "next/server";
import { env } from "~/env";
import { checkAndUpdateQuota } from "~/lib/quota";
import { db } from "~/server/db";

interface KeyRequest {
  key: string;
}

interface AnalysisResult {
  // the structure of the analysis result
  sentiment: string;
  confidence: number;
  [key: string]: unknown;
}

/**
 * Handles sentiment inference requests.
 * @param req The request object.
 * @returns A promise that resolves to a NextResponse object.
 */
export async function POST(req: Request): Promise<NextResponse> {
  try {
    // Get API key from the header
    const apiKey = req.headers.get("Authorization")?.replace("Bearer ", "");
    if (!apiKey) {
      return NextResponse.json({ error: "API key required" }, { status: 401 });
    }

    // Find the user by API key
    const quota = await db.apiQuota.findUnique({
      where: {
        secretKey: apiKey,
      },
      select: {
        userId: true,
      },
    });

    if (!quota) {
      return NextResponse.json({ error: "Invalid API key" }, { status: 401 });
    }

    const { key } = await req.json() as KeyRequest;

    if (!key) {
      return NextResponse.json({ error: "Key is required" }, { status: 400 });
    }

    const file = await db.videoFile.findUnique({
      where: { key },
      select: { userId: true, analyzed: true },
    });

    if (!file) {
      return NextResponse.json({ error: "File not found" }, { status: 404 });
    }

    if (file.userId !== quota.userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 403 });
    }

    if (file.analyzed) {
      return NextResponse.json(
        { error: "File already analyzed" },
        { status: 400 },
      );
    }

    const hasQuota = await checkAndUpdateQuota(quota.userId, true);

    if (!hasQuota) {
      return NextResponse.json(
        { error: "Monthly quota exceeded" },
        { status: 429 },
      );
    }

    // Call sagemaker endpoint
    const sagemakerClient = new SageMakerRuntimeClient({
      region: env.AWS_REGION,
      credentials: {
        accessKeyId: env.AWS_ACCESS_KEY_ID,
        secretAccessKey: env.AWS_SECRET_ACCESS_KEY,
      },
    });

    const command = new InvokeEndpointCommand({
      EndpointName: env.AWS_ENDPOINT_NAME,
      ContentType: "application/json",
      Body: JSON.stringify({
        video_path: `s3://your-bucket-name/${key}`,
      }),
    });

    const response = await sagemakerClient.send(command);
    
    // Safely handle the binary response and convert to text
    const decoder = new TextDecoder();
    const responseBody = response.Body ? decoder.decode(response.Body) : '{}';
    const analysis = JSON.parse(responseBody) as AnalysisResult;

    await db.videoFile.update({
      where: { key },
      data: {
        analyzed: true,
      },
    });

    return NextResponse.json({
      analysis,
    });
  } catch (error: unknown) {
    console.error("Analysis error: ", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 },
    );
  }
}
