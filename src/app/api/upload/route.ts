// import { NextResponse } from "next/server";
// import path from "path";
// import { writeFile } from "fs/promises";
// import pdfParse from "pdf-parse";
// export const runtime = "nodejs";

// export const POST = async (req: Request) => {
//   const formData = await req.formData();
//   const file = formData.get("file");

//   if (!file) {
//     return NextResponse.json({ error: "No files received." }, { status: 400 });
//   }

//   if (!(file instanceof File)) {
//     return NextResponse.json({ error: "Invalid file type." }, { status: 400 });
//   }
//   const buffer = Buffer.from(await file.arrayBuffer());
//   const filename = file.name.replaceAll(" ", "_");
//   if (!filename.toLowerCase().endsWith(".pdf")) {
//     return NextResponse.json(
//       { error: "Invalid file type. Only PDFs are allowed." },
//       { status: 400 }
//     );
//   }

//   try {
//     await writeFile(
//       path.join(process.cwd(), "public/uploads/" + filename),
//       buffer
//     );

//     // return NextResponse.json({ Message: "Success", status: 201 });
//     //     // Extract text from the PDF
//     //     const data = await pdfParse(buffer);

//     // Extract text from the PDF
//     const data = await pdfParse(buffer);

//     console.log("Data", data);

//     return NextResponse.json({
//       message: "Success",
//       // text: data.text,
//       status: 201,
//     });
//   } catch (error) {
//     console.log("Error occurred ", error);
//     return NextResponse.json({ Message: "Failed", status: 500 });
//   }
// };

// import { NextResponse } from "next/server";
// import path from "path";
// import { writeFile } from "fs/promises";

// export const POST = async (req: Request) => {
//   const formData = await req.formData();
//   const file = formData.get("file");

//   if (!file) {
//     return NextResponse.json({ error: "No files received." }, { status: 400 });
//   }

//   if (!(file instanceof File)) {
//     return NextResponse.json({ error: "Invalid file type." }, { status: 400 });
//   }
//   const buffer = Buffer.from(await file.arrayBuffer());
//   const filename = file.name.replaceAll(" ", "_");

//   try {
//     await writeFile(
//       path.join(process.cwd(), "public/uploads/" + filename),
//       buffer
//     );

//     return NextResponse.json({ Message: "Success", status: 201 });
//   } catch (error) {
//     console.log("Error occurred ", error);
//     return NextResponse.json({ Message: "Failed", status: 500 });
//   }
// };

import { NextResponse } from "next/server";
import path from "path";
import pdfParse from "pdf-parse";
import { writeFile, readdir, unlink } from "fs/promises";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

export const POST = async (req: Request) => {
  const formData = await req.formData();
  const file = formData.get("file");

  if (!file) {
    return NextResponse.json({ error: "No file received." }, { status: 400 });
  }

  if (!(file instanceof File)) {
    return NextResponse.json({ error: "Invalid file type." }, { status: 400 });
  }

  const filename = file.name.replaceAll(" ", "_");

  // Check if the file is a PDF based on the file extension.
  if (!filename.toLowerCase().endsWith(".pdf")) {
    return NextResponse.json(
      { error: "Invalid file type. Only PDFs are allowed." },
      { status: 400 }
    );
  }

  const buffer = Buffer.from(await file.arrayBuffer());

  try {
    // Optionally save the PDF file on disk
    const filePath = path.join(process.cwd(), "public/uploads", filename);
    const uploadDir = path.join(process.cwd(), "public/uploads");

    // Remove all existing files in the uploads directory
    const files = await readdir(uploadDir);
    await Promise.all(files.map(file => unlink(path.join(uploadDir, file))));

    await writeFile(filePath, buffer);


    const singleDocPerFileLoader = new PDFLoader(filePath, {
      splitPages: false,
    });

    const singleDoc = await singleDocPerFileLoader.load();
    console.log("singleDoc[0].pageContent.slice(0, 100)", singleDoc[0].pageContent);
    // console.log(singleDoc[0].pageContent.slice(0, 100));

    // Extract text from the PDF
    // const data = await pdfParse(buffer);

    return NextResponse.json({
      message: "Success",
      text: singleDoc[0].pageContent,
      status: 201,
    });
  } catch (error) {
    console.error("Error occurred", error);
    return NextResponse.json(
      { error: "Failed to process the PDF.", status: 500 },
      { status: 500 }
    );
  }
};

