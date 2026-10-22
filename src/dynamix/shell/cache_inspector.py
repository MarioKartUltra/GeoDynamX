# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""An on-demand view of cache ownership, disk traffic and memory pressure."""
from __future__ import annotations

import datetime

from PySide6 import QtCore, QtGui, QtWidgets

from dynamix.engine.cache import Cache
from dynamix.shell.window_fit import TOOL_WINDOW, fit_to_screen, remember_geometry, show_while_inactive


def _bytes(value):
    return f'{value / 1024**3:.2f} GiB' if value >= 1024**3 else f'{value / 1024**2:.1f} MiB'


class _Completion(QtCore.QObject):
    finished = QtCore.Signal(str)


class _Action(QtCore.QRunnable):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation
        self.signals = _Completion()

    def run(self):
        try:
            self.operation()
            message = ''
        except Exception as exc:
            message = f'{type(exc).__name__}: {exc}'
        self.signals.finished.emit(message)


class CacheInspector(QtWidgets.QWidget):
    def __init__(self, cache: Cache, holders=None, parent=None):
        super().__init__(parent, TOOL_WINDOW)
        self.cache = cache
        self.holders = holders or (lambda snapshot: {})
        self._rows = []
        self._busy = False
        self.setWindowTitle('Cache Inspector')
        show_while_inactive(self)
        layout = QtWidgets.QVBoxLayout(self)
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.table = QtWidgets.QTableWidget(0,8)
        self.table.setHorizontalHeaderLabels(['Entry','Kind','Tier','Size','Held by','Last touched','Hits','Key'])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(0,QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._buttons)
        layout.addWidget(self.table,3)
        actions = QtWidgets.QHBoxLayout()
        self.drop = QtWidgets.QPushButton('Drop selected')
        self.spill = QtWidgets.QPushButton('Spill selected')
        self.load = QtWidgets.QPushButton('Load selected')
        self.drop.clicked.connect(self._drop)
        self.spill.clicked.connect(self._spill)
        self.load.clicked.connect(self._load)
        for button in (self.drop,self.spill,self.load):actions.addWidget(button)
        self.reveal = QtWidgets.QPushButton('Reveal disk folder')
        self.reveal.clicked.connect(self._reveal)
        actions.addWidget(self.reveal)
        layout.addLayout(actions)
        controls = QtWidgets.QHBoxLayout()
        self.clear_ram = QtWidgets.QPushButton('Clear unpinned RAM…')
        self.clear_disk = QtWidgets.QPushButton('Clear disk…')
        self.clear_stages = QtWidgets.QPushButton('Drop stages')
        self.trim_gpu = QtWidgets.QPushButton('Trim GPU cache')
        self.clear_ram.clicked.connect(self._clear_ram)
        self.clear_disk.clicked.connect(self._clear_disk)
        self.clear_stages.clicked.connect(self._clear_stages)
        self.trim_gpu.clicked.connect(self._trim_gpu)
        for button in (self.clear_ram,self.clear_disk,self.clear_stages,self.trim_gpu):controls.addWidget(button)
        layout.addLayout(controls)
        log_controls = QtWidgets.QHBoxLayout()
        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText('Filter cache events')
        self.filter.textChanged.connect(self.refresh)
        log_controls.addWidget(self.filter)
        self.pause = QtWidgets.QCheckBox('Pause log')
        log_controls.addWidget(self.pause)
        copy = QtWidgets.QPushButton('Copy log')
        copy.clicked.connect(self._copy)
        log_controls.addWidget(copy)
        layout.addLayout(log_controls)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log,2)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.refresh)
        fit_to_screen(self,1100,740)
        remember_geometry(self,'cache_inspector')
        self.refresh()

    def showEvent(self,event):
        super().showEvent(event)
        self.refresh()
        self.timer.start()

    def hideEvent(self,event):
        self.timer.stop()
        super().hideEvent(event)

    def _selected(self):
        return [self._rows[index.row()] for index in self.table.selectionModel().selectedRows()
                if index.row()<len(self._rows)]

    def _buttons(self):
        rows = self._selected()
        pinned = any('pinned' in row['holders'] for row in rows)
        self.drop.setEnabled(bool(rows) and not pinned and not self._busy)
        self.drop.setToolTip('Release Freeze or the marked level in its owning layer first.' if pinned else '')
        self.spill.setEnabled(bool(rows) and all(row['can_spill'] for row in rows) and not self._busy)
        self.load.setEnabled(bool(rows) and any(row['tier']=='disk' for row in rows) and not self._busy)

    def refresh(self,*unused):
        snapshot = self.cache.snapshot()
        owners = self.holders(snapshot)
        selected = {row['key'] for row in self._selected()}
        self._rows = snapshot['entries']
        self.table.blockSignals(True)
        self.table.setRowCount(len(self._rows))
        self.table.clearSelection()
        for i,row in enumerate(self._rows):
            held = owners.get(row['key'],row['holders'])
            if isinstance(held,str):held=[held]
            label = row.get('label',row['key'])+(' · writing' if row.get('pending') else '')
            when = datetime.datetime.fromtimestamp(row.get('touched',0)).strftime('%H:%M:%S')
            values = [label,row['kind'],row['tier'],_bytes(row['bytes']),', '.join(held),when,
                      str(row.get('hits',0)),row['key'][:8]]
            for col,value in enumerate(values):
                item=QtWidgets.QTableWidgetItem(value)
                item.setToolTip(row['key'] if col==7 else value)
                self.table.setItem(i,col,item)
            if row['key'] in selected:
                self.table.selectionModel().select(self.table.model().index(i,0),
                    QtCore.QItemSelectionModel.SelectionFlag.Select | QtCore.QItemSelectionModel.SelectionFlag.Rows)
        self.table.blockSignals(False)
        report,disk=snapshot['report'],snapshot['disk']
        ram,stages=report['results'],report['stages']
        import psutil
        available=psutil.virtual_memory().available
        rss=psutil.Process().memory_info().rss
        mlx=report['mlx'] or {'active':0,'cache':0}
        self.summary.setText(
            f"RAM {_bytes(ram['bytes'])} / {_bytes(ram['max_bytes'] or 0)} · analyses {ram['analysis']} / {ram['max_results']}"
            f" · pinned {_bytes(ram['pinned_bytes'])}\n"
            f"Disk {_bytes(disk['bytes'])} / {_bytes(disk['max_bytes'])} · {disk['pending']} pending writes · {disk['folder']}\n"
            f"Stages {_bytes(stages['bytes'])} · GPU active {_bytes(mlx['active'])}, cached {_bytes(mlx['cache'])}"
            f" · process {_bytes(rss)} · available {_bytes(available)} (reserve {_bytes(snapshot['pressure_floor'])})"
            f" · hits {snapshot['hits']}, misses {snapshot['misses']}")
        if not self.pause.isChecked():
            query=self.filter.text().casefold()
            events=[]
            for event in reversed(snapshot['events']):
                stamp=datetime.datetime.fromtimestamp(event['time']).strftime('%H:%M:%S.%f')[:-3]
                events.append(f"{stamp}  {event['event']}  {event['label']}  "
                              f"{_bytes(event['bytes'])}  {event['ms']:.1f} ms  "
                              f"{event['key'][:8]}  [{event['thread']}]  {event['detail']}")
            self.log.setPlainText('\n'.join(line for line in events if query in line.casefold()))
        self.reveal.setEnabled(bool(disk['folder']))
        self._buttons()

    def _confirm(self,text):
        return QtWidgets.QMessageBox.question(self,'Cache Inspector',text)==QtWidgets.QMessageBox.StandardButton.Yes

    def _run(self,operation):
        if self._busy:return
        self._busy=True
        self.status.setText('Updating cache…')
        self._buttons()
        task=_Action(operation)
        task.signals.finished.connect(self._finished)
        QtCore.QThreadPool.globalInstance().start(task)

    def _finished(self,message):
        self._busy=False
        self.status.setText(message or 'Cache updated.')
        self.refresh()

    def _drop(self):
        rows=self._selected()
        if any('pinned' in row['holders'] for row in rows):return
        if any('shown' in row['holders'] for row in rows) and not self._confirm(
                'Drop the shown analysis? Its layer will recompute on its next resolve.'):return
        cache=self.cache;keys=[row['key'] for row in rows]
        self._run(lambda:[cache.invalidate(key) for key in keys])

    def _spill(self):
        cache=self.cache;keys=[row['key'] for row in self._selected()]
        self._run(lambda:[cache.spill(key) for key in keys])

    def _load(self):
        cache=self.cache;keys=[row['key'] for row in self._selected() if row['tier']=='disk']
        self._run(lambda:[cache.load(key) for key in keys])

    def _clear_ram(self):
        if self._confirm('Clear every unpinned entry in RAM? Layers may recompute on their next resolve.'):
            self._run(self.cache.clear_unpinned)

    def _clear_disk(self):
        if self._confirm('Delete every entry in the disk tier?'):
            self._run(self.cache.clear_disk)

    def _clear_stages(self):
        self._run(self.cache.stages.clear)

    def _trim_gpu(self):
        self._run(self.cache.trim_gpu)

    def _reveal(self):
        folder=self.cache.snapshot()['disk']['folder']
        if folder:QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(folder))

    def _copy(self):
        QtWidgets.QApplication.clipboard().setText(self.log.toPlainText())
