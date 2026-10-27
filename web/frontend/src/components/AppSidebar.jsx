import React from 'react';
import { ArrowUpRight, Eraser, Layers, LoaderCircle, ScanSearch, SlidersHorizontal } from 'lucide-react';

export function AppSidebar({ selectedTiles, selectedModel, onSelectModel, threshold, onChangeThreshold, maskOpacity, onChangeMaskOpacity, uncertaintyOpacity, onChangeUncertaintyOpacity, useMcDropout, onToggleMcDropout, mcSamples, onChangeMcSamples, isProcessing, onRunDetection, onClearSelection, onClearMasks, hasMasks, results, showUncertainty, onToggleUncertainty }) {
  return <aside className="analysis-panel" aria-label="Configurar análise">
    <div className="panel-heading"><h2>Análise florestal <ArrowUpRight size={20} /></h2><p>Inferência com Sentinel-2 e NIR real.</p></div>
    <div className="panel-scroll">
      <section><h3><span>01</span> Área de interesse</h3>
        <div className={`selection-card ${selectedTiles.length ? 'has-selection' : ''}`}><Layers size={20} /><div><strong>{selectedTiles.length ? `${selectedTiles.length} GeoTIFF(s) selecionado(s)` : 'Comece pelo mapa'}</strong><p>{selectedTiles.length ? selectedTiles.slice(0, 2).map((tile) => tile.file).join(' · ') : 'Clique em células Sentinel-2 para selecionar uma ou mais áreas.'}</p></div></div>
        {selectedTiles.length > 0 && <button className="text-button" disabled={isProcessing} onClick={onClearSelection}><Eraser size={13} />Limpar seleção</button>}
        {hasMasks && <button className="text-button" disabled={isProcessing} onClick={onClearMasks}><Eraser size={13} />Limpar máscaras</button>}
      </section>
      <section><h3><span>02</span> Modelo</h3><div className="model-options">{[['unet', 'U-Net', 'Arquitetura clássica'], ['attn_unet', 'Attention U-Net', 'Mecanismo de atenção']].map(([value, title, description]) => <button key={value} aria-pressed={selectedModel === value} disabled={isProcessing} onClick={() => onSelectModel(value)}><span className="radio-dot" /><div><strong>{title}</strong><small>{description}</small></div></button>)}</div></section>
      <section><h3><span>03</span> Parâmetros <SlidersHorizontal size={14} /></h3>
        <label className="range-label" htmlFor="threshold">Limiar de segmentação <output>{threshold.toFixed(2)}</output></label><input id="threshold" type="range" min="0" max="100" value={threshold * 100} disabled={isProcessing} onChange={event => onChangeThreshold(Number(event.target.value) / 100)} /><div className="range-hints"><span>Mais sensível</span><span>Mais restritivo</span></div>
        <label className="range-label" htmlFor="opacity">Opacidade da máscara <output>{Math.round(maskOpacity * 100)}%</output></label><input id="opacity" type="range" min="0" max="100" value={maskOpacity * 100} onChange={event => onChangeMaskOpacity(Number(event.target.value) / 100)} />
        <label className="check-label"><input type="checkbox" checked={useMcDropout} disabled={isProcessing} onChange={event => onToggleMcDropout(event.target.checked)} /><span>Estimar variabilidade<small>Monte Carlo Dropout</small></span></label>
        {useMcDropout && <><label className="range-label" htmlFor="samples">Amostras <output>{mcSamples}</output></label><input id="samples" type="range" min="2" max="15" value={mcSamples} disabled={isProcessing} onChange={event => onChangeMcSamples(Number(event.target.value))} /></>}
        {results?.overlays?.some((overlay) => overlay.uncertainty_png_base64) && <><label className="range-label" htmlFor="uncertainty-opacity">Opacidade do MC Dropout <output>{Math.round(uncertaintyOpacity * 100)}%</output></label><input id="uncertainty-opacity" type="range" min="0" max="100" value={uncertaintyOpacity * 100} onChange={event => onChangeUncertaintyOpacity(Number(event.target.value) / 100)} /></>}
      </section>
      {results && <section aria-live="polite"><h3>Resultado da análise</h3><div className="result-grid"><div><small>Não-floresta estimada</small><strong>{results.stats.non_forest_pct}%</strong></div><div><small>Floresta</small><strong>{results.stats.forest_pct}%</strong></div></div><p className="result-source">{results.stats.processed_tiles} GeoTIFF(s) processado(s) individualmente · Sentinel-2 · B8/NIR real</p>{results.overlays?.some((overlay) => overlay.uncertainty_png_base64) && <label className="check-label"><input type="checkbox" checked={showUncertainty} onChange={event => onToggleUncertainty(event.target.checked)} />Mostrar mapa de variabilidade</label>}</section>}
    </div>
    <div className="panel-action"><button className="primary-button full" disabled={!selectedTiles.length || isProcessing} onClick={onRunDetection}>{isProcessing ? <LoaderCircle className="spin" size={17} /> : <ScanSearch size={17} />}{isProcessing ? 'Analisando área…' : 'Segmentar cobertura'}</button><p>{selectedTiles.length ? `Inferência sobre ${selectedTiles.length} tile(s) Sentinel-2` : 'Selecione uma ou mais áreas para iniciar'}</p></div>
  </aside>;
}
