import React from 'react';
import { Download } from 'lucide-react';
import JSZip from 'jszip';
import { saveAs } from 'file-saver';

const DownloadButton: React.FC = () => {
  const generateEPUB = async () => {
    const zip = new JSZip();

    // Add mimetype file
    zip.file('mimetype', 'application/epub+zip');

    // Add META-INF directory
    const metaInf = zip.folder('META-INF');
    metaInf.file('container.xml', `<?xml version="1.0" encoding="UTF-8"?>
    <container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
      <rootfiles>
        <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
      </rootfiles>
    </container>`);

    // Add OEBPS directory
    const oebps = zip.folder('OEBPS');

    // Add content.opf file
    oebps.file('content.opf', generateContentOPF());

    // Add toc.ncx file
    oebps.file('toc.ncx', generateTOC());

    // Add chapters
    const sections = [
      // ... (keep the existing sections array)
    ];

    sections.forEach((section, index) => {
      oebps.file(`chapter${index + 1}.xhtml`, generateChapter(section.title, section.content, section.bullets));
    });

    // Generate EPUB file
    const content = await zip.generateAsync({ type: 'blob' });
    saveAs(content, 'SLM_Development_Guide.epub');
  };

  const generateContentOPF = () => {
    // Generate the content.opf file content
    // This file contains metadata about the eBook and its structure
    // You'll need to list all chapters and other content files here
  };

  const generateTOC = () => {
    // Generate the toc.ncx file content
    // This file contains the table of contents for the eBook
    // You'll need to list all chapters and their respective files here
  };

  const generateChapter = (title: string, content: string, bullets: string[]) => {
    return `<?xml version="1.0" encoding="utf-8"?>
    <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
    <html xmlns="http://www.w3.org/1999/xhtml">
    <head>
      <title>${title}</title>
    </head>
    <body>
      <h1>${title}</h1>
      <p>${content}</p>
      <ul>
        ${bullets.map(bullet => `<li>${bullet}</li>`).join('')}
      </ul>
    </body>
    </html>`;
  };

  return (
    <button
      onClick={generateEPUB}
      className="fixed bottom-4 right-4 bg-purple-500 hover:bg-purple-600 text-white font-bold py-2 px-4 rounded-full shadow-lg flex items-center"
    >
      <Download size={20} className="mr-2" />
      Download eBook
    </button>
  );
};

export default DownloadButton;