#! /usr/bin/env node
"use strict";

import fetch from "node-fetch";
import path from "path";
import fs from "fs";
import minimist from "minimist";
import { mkConfig, generateCsv, asString } from "export-to-csv";
import { writeFile } from "node:fs";
import { Buffer } from "node:buffer";

var args = minimist(process.argv.slice(2));
const exportFileName = args["o"] || args["output"] || "image_descriptions.csv";
const imageExtensions = [".jpg", ".JPG", ".png"];
const dir = args["d"] || args["dir"] || "./images";
const waitTime = args["w"] || args["wait"] || 30000; 

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const data = [];

if (Object.keys(args).length === 1 || (args["h"] || args["help"])) {
  console.log(`
  Usage: image-description [options]
  Options:
    -d, --dir <directory>   Directory containing images (default: ./images)
    -o, --output <file>     Output CSV file name (default: image_descriptions.csv)
    -w, --wait <ms>         Wait time between requests in milliseconds (default: 30000)
    -h, --help              Show this help message
    
  Example: image-description -d ./images -o descriptions.csv -w 15000
  Repository: https://github.com/leecheeyong/image-description
    `);
  process.exit(1);
}

if(!fs.existsSync(dir)) {
  console.error(`Directory ${dir} does not exist.`);
  process.exit(1);
}

function asyncFunction(item) {
  return new Promise(async (resolve, reject) => {
    const fullPath = path.join(dir, item.name);
    if (item.isFile()) {
      const ext = path.extname(item.name);
      if (imageExtensions.includes(ext)) {
        await fetch("https://docsbot.ai/api/tools/image-prompter", {
          credentials: "omit",
          headers: {
            "User-Agent":
              "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:139.0) Gecko/20100101 Firefox/139.0",
            Accept: "*/*",
            "Accept-Language": "en-US,en;q=0.5",
            "Content-Type": "application/json",
            "Sec-GPC": "1",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            Priority: "u=0",
          },
          referrer: "https://docsbot.ai/tools/image/description-generator",
          body:
            `{\"type\":\"description\",\"image\":"` +
            fs.readFileSync(`${fullPath}`, "base64") +
            `"}`,
          method: "POST",
          mode: "cors",
        })
          .then((r) => r.text())
          .then((description) => {
            data.push({
              fileName: item.name,
              description: description.trim(),
            });
            resolve();
            console.log(`Description generated for`, item.name)
          });
      }
    }
  });
}

fs.readdir(dir, { withFileTypes: true }, async (err, items) => {
  if (err) {
    console.error(`Error reading directory ${dir}:`, err);
    return;
  }
  for (const item of items) {
    await asyncFunction(item);
    await wait(waitTime);
  }
  const csv = generateCsv(mkConfig({ useKeysAsHeaders: true }))(data);
  writeFile(exportFileName, new Uint8Array(Buffer.from(asString(csv))), (err) => {
    if (err) throw err;
    console.log("CSV file saved: ", exportFileName);
  });
});
