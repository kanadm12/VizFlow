/**
 * Export visualization as PNG image
 */

export const exportSVGToPNG = async (svgElement, fileName = 'flowchart.png') => {
  try {
    if (!svgElement) {
      throw new Error('SVG element not found');
    }

    // Get SVG dimensions
    const svgRect = svgElement.getBoundingClientRect();
    const width = svgRect.width;
    const height = svgRect.height;

    // Clone SVG for manipulation
    const clonedSvg = svgElement.cloneNode(true);
    clonedSvg.setAttribute('width', width);
    clonedSvg.setAttribute('height', height);

    // Convert SVG to Canvas
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext('2d');
    const image = new Image();

    // Convert SVG to data URL
    const svgData = new XMLSerializer().serializeToString(clonedSvg);
    const svg = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(svg);

    return new Promise((resolve, reject) => {
      image.onload = () => {
        // Draw white background
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, width, height);

        // Draw image
        ctx.drawImage(image, 0, 0);
        URL.revokeObjectURL(url);

        // Convert canvas to blob and download
        canvas.toBlob((blob) => {
          const downloadUrl = URL.createObjectURL(blob);
          const link = document.createElement('a');
          link.href = downloadUrl;
          link.download = fileName;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          URL.revokeObjectURL(downloadUrl);
          resolve('✅ Flowchart exported successfully!');
        }, 'image/png');
      };

      image.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error('Failed to load image'));
      };

      image.src = url;
    });
  } catch (error) {
    throw new Error(`PNG export failed: ${error.message}`);
  }
};

/**
 * Export visualization as SVG
 */
export const exportAsSVG = (svgElement, fileName = 'flowchart.svg') => {
  try {
    if (!svgElement) {
      throw new Error('SVG element not found');
    }

    const svgData = new XMLSerializer().serializeToString(svgElement);
    const svg = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(svg);

    const link = document.createElement('a');
    link.href = url;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    return '✅ SVG exported successfully!';
  } catch (error) {
    throw new Error(`SVG export failed: ${error.message}`);
  }
};

/**
 * Copy SVG visualization to clipboard
 */
export const copyToClipboard = async (svgElement) => {
  try {
    if (!svgElement) {
      throw new Error('SVG element not found');
    }

    const canvas = document.createElement('canvas');
    const svgRect = svgElement.getBoundingClientRect();
    canvas.width = svgRect.width;
    canvas.height = svgRect.height;

    const ctx = canvas.getContext('2d');
    const image = new Image();

    const svgData = new XMLSerializer().serializeToString(svgElement);
    const svg = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(svg);

    return new Promise((resolve, reject) => {
      image.onload = () => {
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(image, 0, 0);
        URL.revokeObjectURL(url);

        canvas.toBlob(async (blob) => {
          try {
            await navigator.clipboard.write([
              new ClipboardItem({ 'image/png': blob })
            ]);
            resolve('✅ Copied to clipboard!');
          } catch (err) {
            reject(new Error('Failed to copy to clipboard'));
          }
        }, 'image/png');
      };

      image.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error('Failed to load image'));
      };

      image.src = url;
    });
  } catch (error) {
    throw new Error(`Clipboard copy failed: ${error.message}`);
  }
};

/**
 * Generate downloadable report with visualization
 */
export const generateReport = async (modelGraph, svgElement, reportFormat = 'html') => {
  try {
    if (!modelGraph || !svgElement) {
      throw new Error('Missing model data or visualization');
    }

    const svgData = new XMLSerializer().serializeToString(svgElement);
    const timestamp = new Date().toLocaleString();

    if (reportFormat === 'html') {
      const htmlContent = `
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="UTF-8">
          <title>VizFlow Model Report</title>
          <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .header { background: linear-gradient(135deg, #3b82f6, #06b6d4); color: white; padding: 20px; border-radius: 8px; }
            .content { background: white; padding: 20px; border-radius: 8px; margin-top: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
            .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 15px 0; }
            .stat-item { background: #f9f9f9; padding: 15px; border-radius: 5px; border-left: 4px solid #3b82f6; }
            .stat-value { font-size: 24px; font-weight: bold; color: #3b82f6; }
            .stat-label { color: #666; font-size: 12px; margin-top: 5px; }
            .visualization { margin: 20px 0; text-align: center; }
            svg { max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }
            .footer { color: #666; font-size: 12px; margin-top: 20px; }
          </style>
        </head>
        <body>
          <div class="header">
            <h1>VizFlow Model Visualization Report</h1>
            <p>Generated on ${timestamp}</p>
          </div>
          <div class="content">
            <h2>Model Statistics</h2>
            <div class="stats">
              <div class="stat-item">
                <div class="stat-value">${modelGraph.layers?.length || 0}</div>
                <div class="stat-label">Total Layers</div>
              </div>
              <div class="stat-item">
                <div class="stat-value">${modelGraph.connections?.length || 0}</div>
                <div class="stat-label">Connections</div>
              </div>
              <div class="stat-item">
                <div class="stat-value">${(modelGraph.totalParams || 0).toLocaleString()}</div>
                <div class="stat-label">Parameters</div>
              </div>
            </div>
            <h2>Model Architecture</h2>
            <div class="visualization">
              ${svgData}
            </div>
            <div class="footer">
              <p>This report was generated by VizFlow - AI/ML Model Visualizer</p>
            </div>
          </div>
        </body>
        </html>
      `;

      const blob = new Blob([htmlContent], { type: 'text/html;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `model-report-${Date.now()}.html`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      return '✅ Report exported as HTML!';
    }
  } catch (error) {
    throw new Error(`Report generation failed: ${error.message}`);
  }
};
